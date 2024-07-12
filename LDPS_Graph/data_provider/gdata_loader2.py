import torch
import numpy as np
import pickle
import pandas as pd
import scipy.sparse as sp
from torch_geometric.data import Data
from utils.timefeatures import time_features

class LargeGraphTemporalLoader(object):
   
    def __init__( self, data_path, edge_path=None, lags = 3, p_len=1,  partition = 8, train=True,timeenc=0):
        
        if 'C2TM' in data_path:
            df = pd.read_csv(data_path)/(8*1024*1024)
            df = df.drop('time',axis=1)
            self._data = df.values
        elif 'CBS' in data_path:
            df = pd.read_csv(data_path)
            df = df.drop('time',axis=1)
            self._data = df.values
        else: 
            self._data = np.load(data_path)[...,2:4]/10
            print(self._data.shape, 'Call ...')
 
        # self.timeenc = timeenc
        # self.data_stamp = self._get_stamp(df[['time']])
        
        self.edge_path = edge_path
        self.lags = lags
        self.p_len = p_len
        self.partition = partition
        self.train = train
        
        

        self.n_node = self._data.shape[1]
        self.times =  self._data.shape[0] - lags - p_len + 1
        print('data shape: ',self._data.shape, self.n_node)
        
        if edge_path is not None:
            with open(edge_path, 'rb') as f:
                self._adj_mx = pickle.load(f)["adj_mx"]
        else:
            self._adj_mx = np.diag([1 for _ in range(self._data.shape[1])])
            print(self._adj_mx.shape, 'adj_max shape ...')

    def _get_stamp(self, df_stamp):
        
        df_stamp['date'] = pd.to_datetime(df_stamp.time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        return data_stamp
    

    def _partition_edges(self,idxes):
        edge_rows = self._adj_mx[idxes]
        edge =  edge_rows[:, idxes]

        # adj = sp.coo_matrix(edge)
        # edge_coo = np.array([[adj.row[i],adj.col[i]] for i in range(len(adj.row))])
        # self.edge_index = torch.LongTensor(edge_coo.T)
        # self.edge_weight = torch.FloatTensor(adj.data)
        # print(self.edge_index.shape,self.edge_weight.shape, '---')

        edge = torch.Tensor(edge)
        spa_edge = edge.to_sparse().coalesce()
        self.edge_index = spa_edge.indices().long()
        self.edge_weight = spa_edge.values().float()


    def _partition_targets_and_features(self,idxes):

        _data = self._data[:,idxes]

        self.features = [
            _data[i : i + self.lags, :].T
            for i in range(self.times)
        ]
        self.targets = [
            _data[i + self.lags: i+ self.lags+ self.p_len, :].T
            for i in range(self.times)
        ]
        
    def _re_partition_graph(self,time_index):
        n_subnode = self.n_node//self.partition
        if self.train:
            idxes = np.random.randint(0, self.n_node, n_subnode)
        else: 
            start = time_index*n_subnode
            end = (time_index+1)*n_subnode
            if end > self.n_node:
                end = self.n_node
                start = end - n_subnode
            idxes = np.arange(start, end)

        self._partition_edges(idxes)
        self._partition_targets_and_features(idxes)

        

    def __getitem__(self, time_index):
        
        if time_index == 0: 
            self._re_partition_graph(time_index)

        x = torch.FloatTensor(self.features[time_index])
        y = torch.FloatTensor(self.targets[time_index])
        snapshot = Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_weight, y=y)
        return snapshot

    def __next__(self):
        if self.t < self.times:
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self
    
    def __len__(self):
        return self.times
