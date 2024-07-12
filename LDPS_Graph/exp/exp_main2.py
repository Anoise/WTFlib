# from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer,PatchTST, FEDformer, \
      Mvstgn, DLinear, Periodformer, DecomLinear, DecomLinearV2
      
from models.stid_arch import stid_arch
from models.dydcrnn_arch import dydcrnn
from models.dydgcrn_arch import dydgcrn
from models.gwnet_arch import gwnet
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from data_provider.gdata_loader2 import LargeGraphTemporalLoader

from thop import clever_format
from thop import profile

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Periodformer': Periodformer,
            'PatchTST': PatchTST,
            'FEDformer':FEDformer,
            'Autoformer': Autoformer,
            'Informer': Informer,
            'Transformer': Transformer,
            'DLinear': DLinear,
            'STID':stid_arch,
            'Mvstgn': Mvstgn,
            'DyDcrnn': dydcrnn,
            'DyDgcrn': dydgcrn,
            'Gwnet': gwnet,
            'DecomLinear': DecomLinear,
            'DecomLinearV2': DecomLinearV2,
        }
        
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, test=False):

        if "C2TM" in self.args.data:
            test_loader =  LargeGraphTemporalLoader(data_path=self.root_path+'/c2tm_bytes_test.csv',edge_path=self.root_path+'/adj_mx_0.75.pkl', lags=self.args.seq_len, p_len = self.args.pred_len, partition=self.args.n_part, train=False)

            if test: return test_loader

            train_loader =  LargeGraphTemporalLoader(data_path=self.root_path+'/c2tm_bytes_train.csv',edge_path=self.root_path+'/adj_mx_0.75.pkl', lags=self.args.seq_len, p_len = self.args.pred_len, partition=self.args.n_part)
            val_loader =  LargeGraphTemporalLoader(data_path=self.root_path+'/c2tm_bytes_val.csv',edge_path=self.root_path+'/adj_mx_0.75.pkl', lags=self.args.seq_len, p_len = self.args.pred_len, partition=self.args.n_part, train=False)
        elif "CBSU" in self.args.data:
            print('load CBSU ...')
            test_loader =  LargeGraphTemporalLoader(data_path=self.root_path+'/tai_g_up_test.csv',edge_path=self.root_path+'/tai_all_adj_mx_0.75.pkl', lags=self.args.seq_len, p_len = self.args.pred_len, partition=self.args.n_part, train=False)

            if test: return test_loader

            train_loader =  LargeGraphTemporalLoader(data_path=self.root_path+'/tai_g_up_train.csv',edge_path=self.root_path+'/tai_all_adj_mx_0.75.pkl', lags=self.args.seq_len, p_len = self.args.pred_len, partition=self.args.n_part)
            val_loader =  LargeGraphTemporalLoader(data_path=self.root_path+'/tai_g_up_val.csv',edge_path=self.root_path+'/tai_all_adj_mx_0.75.pkl', lags=self.args.seq_len, p_len = self.args.pred_len, partition=self.args.n_part, train=False)
        elif "CBS" in self.args.data:
            test_loader =  LargeGraphTemporalLoader(data_path=self.root_path+'/tai_g_down_test.csv',edge_path=self.root_path+'/tai_all_adj_mx_0.75.pkl', lags=self.args.seq_len, p_len = self.args.pred_len, partition=self.args.n_part, train=False)

            if test: return test_loader

            train_loader =  LargeGraphTemporalLoader(data_path=self.root_path+'/tai_g_down_train.csv',edge_path=self.root_path+'/tai_all_adj_mx_0.75.pkl', lags=self.args.seq_len, p_len = self.args.pred_len, partition=self.args.n_part)
            val_loader =  LargeGraphTemporalLoader(data_path=self.root_path+'/tai_g_down_val.csv',edge_path=self.root_path+'/tai_all_adj_mx_0.75.pkl', lags=self.args.seq_len, p_len = self.args.pred_len, partition=self.args.n_part, train=False)
        elif "Milano" in self.args.data:
            test_loader =  LargeGraphTemporalLoader(data_path=self.root_path+'/test.npy', edge_path=None, lags=self.args.seq_len, p_len = self.args.pred_len, partition=self.args.n_part, train=False)

            if test: return test_loader

            train_loader =  LargeGraphTemporalLoader(data_path=self.root_path+'/train.npy', edge_path=None, lags=self.args.seq_len, p_len = self.args.pred_len, partition=self.args.n_part)
            val_loader =  LargeGraphTemporalLoader(data_path=self.root_path+'/val.npy', edge_path=None, lags=self.args.seq_len, p_len = self.args.pred_len, partition=self.args.n_part, train=False)
        else:
            raise "Error ..."
        
        return train_loader, val_loader, test_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), 
                                 lr=self.args.learning_rate, 
                                 weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        if self.args.loss=='mse':
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, snapshot in enumerate(vali_loader):
                batch_x = snapshot.x.to(self.device)
                edge_index = snapshot.edge_index.cuda()
                edge_attr = snapshot.edge_attr.cuda()
                batch_y = snapshot.y.to(self.device)

                # encoder - decoder
                
                if 'Decom' in self.args.model:
                    outputs = self.model(batch_x, edge_index, edge_attr)
                elif 'TST' in self.args.model:
                    outputs = self.model(batch_x, edge_index, edge_attr)
                elif 'former' in self.args.model \
                      or 'Linear' in self.args.model:
                    if len(batch_x.shape)<3:
                        batch_x = batch_x.unsqueeze(0)
                        batch_y = batch_y.unsqueeze(0)
                    batch_x = batch_x.permute(0,2,1)
                    batch_y = batch_y.permute(0,2,1)
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    outputs = self.model(batch_x, dec_inp)
                elif 'rnn' in self.args.model:
                    if len(batch_x.shape)<3:
                        batch_x = batch_x.unsqueeze(0)
                        batch_y = batch_y.unsqueeze(0)
                    batch_x = batch_x.permute(0,2,1)
                    batch_y = batch_y.permute(0,2,1)
                    outputs = self.model(batch_x, edge_index, edge_attr)
                else:
                    if len(batch_x.shape)<3:
                        batch_x = batch_x.unsqueeze(0)
                        batch_y = batch_y.unsqueeze(0)
                    batch_x = batch_x.permute(0,2,1)
                    batch_y = batch_y.permute(0,2,1)
                    batch_x = batch_x.unsqueeze(-1)
                    batch_y = batch_y.unsqueeze(-1)
                    outputs = self.model(batch_x, edge_index, edge_attr)
                    
                batch_y = batch_y.to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_loader, vali_loader, test_loader = self._get_data()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        if self.args.lradj == 'Mvstgn':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optim, milestones=[0.5 * self.args.train_epochs, 0.75 * self.args.train_epochs], gamma=0.1)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                                steps_per_epoch = train_steps,
                                                pct_start = self.args.pct_start,
                                                epochs = self.args.train_epochs,
                                                max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, snapshot in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = snapshot.x.to(self.device)
                edge_index = snapshot.edge_index.cuda()
                edge_attr = snapshot.edge_attr.cuda()
                batch_y = snapshot.y.to(self.device)
                loss = None
                # encoder - decoder
                if 'Decom' in self.args.model:
                    outputs, loss = self.model(batch_x, edge_index, edge_attr, batch_y)
                elif 'TST' in self.args.model:
                    outputs = self.model(batch_x, edge_index, edge_attr)
                elif 'former' in self.args.model \
                      or 'Linear' in self.args.model:
                    if len(batch_x.shape)<3:
                        batch_x = batch_x.unsqueeze(0)
                        batch_y = batch_y.unsqueeze(0)
                    batch_x = batch_x.permute(0,2,1)
                    batch_y = batch_y.permute(0,2,1)
                    if "Period" in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                        outputs = self.model(batch_x, dec_inp)
                elif 'rnn' in self.args.model:
                    if len(batch_x.shape)<3:
                        batch_x = batch_x.unsqueeze(0)
                        batch_y = batch_y.unsqueeze(0)
                    batch_x = batch_x.permute(0,2,1)
                    batch_y = batch_y.permute(0,2,1)
                    outputs = self.model(batch_x, edge_index, edge_attr, batch_y)
                else:
                    if len(batch_x.shape)<3:
                        batch_x = batch_x.unsqueeze(0)
                        batch_y = batch_y.unsqueeze(0)
                    batch_x = batch_x.permute(0,2,1)
                    batch_y = batch_y.permute(0,2,1)
                    batch_x = batch_x.unsqueeze(-1)
                    batch_y = batch_y.unsqueeze(-1)
                    if self.args.model in ('STID','Mvstgn'):
                        outputs = self.model(batch_x, edge_index, edge_attr)
                    else:
                        outputs = self.model(batch_x, edge_index, edge_attr, ycl=None, batch_seen=i)

                
                #outputs = outputs[-self.args.pred_len:]
                #batch_y = batch_y.to(self.device)
                if loss is None:
                    loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj in ('TST', 'Mvstgn'):
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj not in ('TST', 'Mvstgn'):
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=1):
        test_loader = self._get_data(test=True)
        
        if test:
            print('loading model.............')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        pds, gts = [], []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, snapshot in enumerate(test_loader):
                batch_x = snapshot.x.to(self.device)
                edge_index = snapshot.edge_index.cuda()
                edge_attr = snapshot.edge_attr.cuda()
                batch_y = snapshot.y.to(self.device)

               
                if 'Decom' in self.args.model:
                    outputs = self.model(batch_x, edge_index, edge_attr)
                elif 'TST' in self.args.model:
                    outputs = self.model(batch_x,edge_index,edge_attr)
                    #outputs = outputs.squeeze(-1)
                elif 'former' in self.args.model \
                      or 'Linear' in self.args.model:
                    if len(batch_x.shape)<3:
                        batch_x = batch_x.unsqueeze(0)
                        batch_y = batch_y.unsqueeze(0)
                    batch_x = batch_x.permute(0,2,1)
                    batch_y = batch_y.permute(0,2,1)
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    outputs = self.model(batch_x, dec_inp)
                elif 'rnn' in self.args.model:
                    if len(batch_x.shape)<3:
                        batch_x = batch_x.unsqueeze(0)
                        batch_y = batch_y.unsqueeze(0)
                    batch_x = batch_x.permute(0,2,1)
                    batch_y = batch_y.permute(0,2,1)
                    outputs = self.model(batch_x, edge_index, edge_attr)
                else:
                    if len(batch_x.shape)<3:
                        batch_x = batch_x.unsqueeze(0)
                        batch_y = batch_y.unsqueeze(0)
                    batch_x = batch_x.permute(0,2,1)
                    batch_y = batch_y.permute(0,2,1)
                    outputs = self.model(batch_x.unsqueeze(-1), edge_index, edge_attr)
                    outputs = outputs.squeeze(-1)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

                input = batch_x.detach().cpu().numpy()
                #print(input.shape, pred.shape, true.shape, 'dddxxx ...')

                if 'Decom' in self.args.model or 'TST' in self.args.model:
                    pd = np.concatenate((input, pred), axis=-1)
                    gt = np.concatenate((input, true), axis=-1)
                else:
                    pd = np.concatenate((input, pred), axis=1)
                    gt = np.concatenate((input, true), axis=1)
                #visual(gt[0,:], pd[0,:], os.path.join(folder_path, str(i) + '.pdf'))
                pds.append(pd)
                gts.append(gt)

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        pds = np.array(pds)
        gts = np.array(gts)

        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'pred.npy', pds)
        np.save(folder_path + 'true.npy', gts)
        

        mae, mse, rmse, mape, mspe, rse, corr = metric(np.array(preds), np.array(trues))
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
