import torch
import torch.nn as nn
from baseline.rank.RankNet.ranknet_model import get_pair_node
from model.SiamesaRank import SiameseNet

class Dir_LSTM_Rank(nn.Module):
    def __init__(self, args):
        super(Dir_LSTM_Rank, self).__init__()
        self.nfeat = args.d_e  # 16  原始维度m
        self.nhid = args.d_h  # 8 嵌入维度
        self.path_length = args.path_length # 5
        self.num_paths = args.num_paths     # 100
        self.mode = 'LSTM'
        self.droprate = args.dropout
        self.dropout = nn.Dropout(p=self.droprate)
        self.nonlinear = nn.Tanh()
        self.num_features = 8 # 8
        self.bn = nn.BatchNorm1d(self.nfeat, track_running_stats=True)
        self.first = nn.Sequential(
            nn.Linear(self.nfeat, self.nhid * 4),
            nn.BatchNorm1d(self.nhid * 4),
            self.nonlinear, # tanh
            self.dropout)

        if self.mode == 'LSTM':
            self.lstm = nn.LSTM(input_size=self.nhid * 4,
                               hidden_size = 2, # 2
                               num_layers=1,
                               # dropout=dropout,
                               bidirectional=True)
                               # batch_first=False,input=(5, batch_size*100)
        elif self.mode == 'GRU':
            self.gru = nn.GRU(input_size=self.nhid * 4,
                              hidden_size= 2,
                              num_layers=1,
                              # dropout=dropout,
                              bidirectional=True)

        self.SiameseNet = SiameseNet(self.num_features)
        # self.SiameseNet = RankNet(args)

    def forward(self, x,batch_node_idx):
        x = self.bn(x)
        x = self.first(x)  # [6000,24]
        x = x.view(-1, self.path_length, x.size(1)).transpose_(0, 1)
        if self.mode == 'LSTM':
            outputs, (ht, ct) = self.lstm(x)
        elif self.mode == 'GRU':
            outputs, ht = self.gru(x)
        _, batch_size, hidlen = outputs.size()

        outputs = outputs.view(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen) # [5,batch_size,100,12]
        outputs = outputs.mean(dim=2)

        selfloop = outputs[0]
        outputs = torch.cat((selfloop, (outputs[1:].mean(dim=0))), dim=1)

        score = self.SiameseNet(outputs)

        idx_1, idx_2 = get_pair_node(batch_node_idx)
        return score ,idx_1,idx_2