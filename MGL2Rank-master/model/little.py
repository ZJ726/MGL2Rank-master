from torch import nn
import torch
from model.SiamesaRank import SiameseNet

class Little(nn.Module):
    def __init__(self, args):
        super(Little, self).__init__()
        self.num_paths = args.num_paths
        self.path_length = args.path_length
        self.nfeat = args.d_e # 9
        self.nhid = args.d_h # 12
        self.mode = 'GRU'
        self.droprate = args.dropout
        self.nonlinear = nn.Tanh()
        self.dropout = nn.Dropout(p=self.droprate)  # p: probability of an element to be zeroed. Default: 0.5
        self.num_features = self.nfeat * 4
        self.bn = nn.BatchNorm1d(self.nfeat, track_running_stats=True)
        self.first = nn.Sequential(
            nn.Linear(self.nfeat, self.nfeat), # 6-6
            nn.BatchNorm1d(self.nfeat),
            self.nonlinear,
            self.dropout)
        if self.mode == 'LSTM':  # 6-6-12
            self.lstm = nn.LSTM(input_size=self.nfeat,
                               hidden_size= 2, #self.nfeat,
                               num_layers=1,
                               # dropout=dropout,
                               bidirectional=True)
        elif self.mode == 'GRU':
            self.gru = nn.GRU(input_size=self.nfeat,
                              hidden_size=self.nfeat,
                              num_layers=1,
                              # dropout=dropout,
                              bidirectional=True)

        self.SiameseNet = SiameseNet(self.num_features)


    def forward(self, src):
        # 1.FNN
        src = self.bn(src)
        src = self.first(src)  # 6-6
        src = src.view(-1, self.path_length, src.size(1)).transpose_(0, 1)  # x=[6000,6]->[1200,5,6]—>[5,1200,6]
        # 2.LSTM
        if self.mode == 'LSTM':  # input:24,output:12
            outputs, (ht, ct) = self.lstm(src)  # output=(5,1200,12)
        elif self.mode == 'GRU':
            outputs, ht = self.gru(src)
        _, batch_size, hidlen = outputs.size()  # 24
        src = outputs.reshape(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen).mean(dim=2)
        selfloop = src[0] # (32,16)
        output = torch.cat((selfloop,(src[1:].mean(dim=0))), dim=1)  # outputs:24
        score = self.SiameseNet(output)
        return score
