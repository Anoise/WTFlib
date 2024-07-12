import torch
from torch import nn
import numpy as np
from torch_geometric.utils import to_dense_adj
from .dydcrnn_cell import DCGRUCell


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, args):
        self.max_diffusion_step = args.max_diffusion_step
        self.cl_decay_steps = args.cl_decay_steps
        self.filter_type = args.filter_type
        self.num_nodes = args.enc_in
        self.num_rnn_layers = args.num_rnn_layers
        self.rnn_units = args.rnn_units
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args)
        self.input_dim = int(args.enc_in)
        self.seq_len = int(args.seq_len)  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj_mx, hidden_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros(
                (self.num_rnn_layers, batch_size, self.hidden_state_size)).to(inputs.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(
                output, hidden_state[layer_num], adj_mx)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        # runs in O(num_layers) so not too slow
        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, args):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, args)
        self.output_dim = 1
        self.horizon = args.pred_len  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj_mx, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(
                output, hidden_state[layer_num], adj_mx)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class Model(nn.Module, Seq2SeqAttrs):
    """
    Paper: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
    Link: https://arxiv.org/abs/1707.01926
    Codes are modified from the official repo:
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_cell.py,
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_model.py
    """

    def __init__(self, args):
        super().__init__()
        Seq2SeqAttrs.__init__(self, args)
        self.encoder_model = EncoderModel(args)
        self.decoder_model = DecoderModel(args)
        self.use_curriculum_learning = args.use_curriculum_learning

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj_mx):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(
                inputs[t], adj_mx, encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj_mx, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros(
            (batch_size, self.num_nodes * self.decoder_model.output_dim)).to(encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input, adj_mx, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, history_data,
                edge_index,
                edge_attr,
                future_data=None,
                batch_seen: int = 1, **kwargs) -> torch.Tensor:
        """Feedforward function of DCRNN.

        Args:
            history_data (torch.Tensor): history data with shape [L, B, N*C]
            future_data (torch.Tensor, optional): future data with shape [L, B, N*C_out]
            batch_seen (int, optional): batches seen till now, used for curriculum learning. Defaults to None.

        Returns:
            torch.Tensor: prediction with shape [L, B, N*C_out]
        """
        adj_mx = to_dense_adj(edge_index, edge_attr=edge_attr)
        adj_mx = adj_mx.reshape(adj_mx.size(1), adj_mx.size(2))
        adj_mx = adj_mx.cpu().detach().numpy()
        # reshape data
        #batch_size, length, num_nodes, channels = history_data.shape
        # history_data = history_data.reshape(batch_size, length, num_nodes * channels)      # [B, L, N*C]
        history_data = history_data.transpose(0, 1)         # [L, B, N*C]

        if future_data is not None:
            # future_data = future_data[..., [0]]     # teacher forcing only use the first dimension.
            #batch_size, length, num_nodes, channels = future_data.shape
            # future_data = future_data.reshape(batch_size, length, num_nodes * channels)      # [B, L, N*C]
            future_data = future_data.transpose(0, 1)         # [L, B, N*C]
        # DCRNN
        encoder_hidden_state = self.encoder(history_data, adj_mx)
        outputs = self.decoder(encoder_hidden_state, adj_mx, future_data,
                               batches_seen=batch_seen)      # [L, B, N*C_out]

        # reshape to B, L, N, C
        L, B, _ = outputs.shape
        outputs = outputs.transpose(0, 1)  # [B, L, N*C_out]
        outputs = outputs.view(B, L, self.num_nodes,
                               self.decoder_model.output_dim)

        if batch_seen == 0:
            print("Warning: decoder only takes the first dimension as groundtruth.")
            print("Parameter Number: ".format(count_parameters(self)))
            print(count_parameters(self))
        return outputs.squeeze(-1)
