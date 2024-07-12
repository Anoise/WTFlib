import torch
import torch.nn.functional as F


def get_edges(x, gama):
    x = F.normalize(x, dim=-1)
    edge = x.matmul(x.T)
    edge[edge< gama] = 0 
    edge[edge>= gama] = 1 
    spa_edge = edge.to_sparse().coalesce()
    indices = spa_edge.indices().long()
    values = spa_edge.values().float()

    # print(edge)
    # print(indices.shape, indices.dtype)
    # print(values.shape, values.dtype)
    # exit()

    return x, indices, values


class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        out_channels= configs.d_model * 2
        self.pred_len = configs.pred_len
        
        self.l_x1 = torch.nn.Linear(configs.seq_len, out_channels)
        self.l_m1 = torch.nn.Linear(1, out_channels//4)
        self.l_s1 = torch.nn.Linear(1, out_channels//4)
        
        self.l_x2 = torch.nn.Linear(out_channels, configs.pred_len)
        self.l_m2 = torch.nn.Linear(out_channels//4, 1)
        self.l_s2 = torch.nn.Linear(out_channels//4, 1)
        
        self.l_o1 = torch.nn.Linear(configs.pred_len, out_channels)
        self.l_o2 = torch.nn.Linear(out_channels, configs.pred_len)


        self.L1 = torch.nn.L1Loss()
        self.L2 = torch.nn.MSELoss()

        print("L_Decom ...")

    def forward(self, x, edge_indexs=None, edge_weights=None, y=None):
        
        x, x_m, x_s = self.decompose(x)


        x1 = self.l_x1(x)
        x1 = F.relu(x1)
        x2 = self.l_x2(x1)

        m1 = self.l_m1(x_m)
        m1 = F.relu(m1)
        m2 = self.l_m2(m1)

        s1 = self.l_s1(x_s)
        s1 = F.relu(s1)
        s2 = self.l_s2(s1)
        
        out = x2 * s2

        out = self.l_o1(out)
        out = F.relu(out)
        out = self.l_o2(out) + m2

        if y is not None:
            
            _y, y_m, y_s = self.decompose(y)

            loss = self.L1(out, y) + 0.1*(self.L2(x2,_y) + self.L2(m2, y_m) + self.L2(s2, y_s))
        
            return out, loss
        
        return out
    
    def decompose(self, y):
        means = y.mean(-1, keepdim=True).detach()
        y = y - means
        stdev = torch.sqrt(torch.var(y, dim=-1, keepdim=True, unbiased=False) + 1e-7)
        y /= stdev
        return y, means, stdev
