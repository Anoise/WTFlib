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

    return x, indices, values


class moving_avg(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding= 0)
        print(self.kernel_size,'kernel_size ...')

    def forward(self, x):
        # padding on the both ends of time series
        if len(x.shape) == 2:
            front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
            end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
            x = torch.cat([front, x, end], dim=1)
            x = self.avg(x)
        else:
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=1)
            x = self.avg(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        return x

class series_decomp(torch.nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, trend_size, seasonal_size):
        super(series_decomp, self).__init__()
        self.trend= moving_avg(trend_size, stride=1)
        self.seasonal = moving_avg(seasonal_size, stride=1)

    def forward(self, x):
        trend = self.trend(x)
        x_ = x - trend
        seasonal = self.seasonal(x_)
        res = x_ - seasonal
        return res, trend, seasonal



class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        out_channels= configs.d_model * 2
        
        self.l_x1 = torch.nn.Linear(configs.seq_len, out_channels)
        self.l_m1 = torch.nn.Linear(configs.seq_len, out_channels//4)
        self.l_s1 = torch.nn.Linear(configs.seq_len, out_channels//4)
        
        self.l_x2 = torch.nn.Linear(out_channels, configs.pred_len)
        self.l_m2 = torch.nn.Linear(out_channels//4, configs.pred_len)
        self.l_s2 = torch.nn.Linear(out_channels//4, configs.pred_len)
        
        self.l_o1 = torch.nn.Linear(configs.pred_len, out_channels)
        self.l_o2 = torch.nn.Linear(out_channels, configs.pred_len)

        self.x_decompose = self.decompose(configs.seq_len)
        self.y_decompose = self.decompose(configs.pred_len)

        self.L1 = torch.nn.L1Loss()
        self.L2 = torch.nn.MSELoss()

        print("L_Decom V2 ...")

    def forward(self, x, edge_indexs=None, edge_weights=None, y=None):
        
        x, x_m, x_s = self.x_decompose(x)

        x1 = self.l_x1(x)
        x1 = F.relu(x1)
        x2 = self.l_x2(x1) 

        m1 = self.l_m1(x_m)
        m1 = F.relu(m1)
        m2 = self.l_m2(m1)

        s1 = self.l_s1(x_s)
        s1 = F.relu(s1)
        s2 = self.l_s2(s1)
        
        out = x2 + s2 

        out = self.l_o1(out)
        out = F.relu(out)
        out = self.l_o2(out) + m2

        if y is not None:
            
            _y, y_m, y_s = self.y_decompose(y)

            loss = self.L1(out, y) + 0.1*(self.L2(x2,_y) + self.L2(m2, y_m) + self.L2(s2, y_s))
        
            return out, loss
        
        return out

    def decompose(self, input_length):

        trend_size = input_length // 4
        if trend_size % 2 == 0:
            trend_size += 1

        ssnl_size = trend_size // 2
        if ssnl_size % 2 == 0:
            ssnl_size += 1

        trend_size = max(3, trend_size)

        ssnl_size = max(3, ssnl_size)

        return series_decomp(trend_size, ssnl_size)