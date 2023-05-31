import torch
import torch.nn as nn
import math
from model.SiamesaRank import SiameseNet
import torch.nn.functional as F
from baseline.rank.RankNet.ranknet_model import RankNet, get_pair_node


class PositionalEncoding(nn.Module):  # 返回一个seq的位置编码(序列个数，最多节点个数,节点维度)
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype = torch.float).unsqueeze(1)  # torch([[0],[1],……,[4999]])
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model)) # torch.Size([400])
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',pe)  # 定义一组参数，模型训练时这组参数不会更新，只可人为地改变值,保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
    def forward(self,x):
        x = x+self.pe[:x.size(0),:]
        return self.dropout(x)  # input = (序列个数，最多节点个数,节点维度),output = (序列个数，最大多节点个数，节点维度)

# 计算attention score=softmax(k*q/根号3)*v
class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None,dropout=None):
        # q、k、v = 5，1，16,200
        dk = query.size()[-1]  # dk = 200
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk) # (q*k)/根号3
        #scores 5*1*16*1

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1) # (5,1,16,1),对最后一维softmax
        if dropout is not None:
            attention = dropout(attention)
        return attention.matmul(value) # (5,1,16,200)*(5,1,16,1)=(5,1,16,200)

# self_atten
class MultiSelfAttention(nn.Module):   # 多头自注意力机制
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model%heads == 0
        self.d_model = d_model  # 200
        self.d_k = d_model // heads  # heads =1头 ,d_k=200
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention() # (5,1,16,200)

    def forward(self, q, k, v, mask=None):   # q=k=v=(5*16*200)
        if mask is not None:
            # 试图将同一个mask用在所有的头上
            mask = mask.unsqueeze(1) # 对mask增一维,如原维度为(2,3)变成(2,1,3)

        bs = q.size(0)  # 5
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)  # k= 5,16,1,200
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)  # q= 5,16,1,200
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)  # q= 5,16,1,200
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2) # k= 5,1,16,200
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        """缩写:等价于上述六行
            query, key, value =[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                                for l, x in zip(self.linears, (query, key, value))]
        """
        # 计算attention score = (5,1,16,200)
        scores = self.attention(q, k, v, mask=mask, dropout=self.dropout)

        # concatenate heads and put through final linear layer，(5,1,16,200)——>(5,16,1,200)——>(5,16,200)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat) # 一层Linear=,作用？

        return output

"""TransLSTM: linear-LSTM-pooling-Linear-Transformer-linear-pooling-concat-排序"""
class TransLSTM_T(nn.Module):
    def __init__(self, args, nhead):
        super(TransLSTM_T, self).__init__()
        self.num_paths = args.num_paths
        self.path_length = args.path_length
        self.nfeat = args.d_e
        self.nhid = args.d_h  #200
        self.droprate = args.dropout
        self.nonlinear = nn.Tanh()
        self.dropout = nn.Dropout(p=self.droprate)   #p: probability of an element to be zeroed. Default: 0.5
        self.activation = F.relu
        self.dim_feedforward = 24
        self.num_features = 24
        self.bn = nn.BatchNorm1d(self.nfeat, track_running_stats=True)
        self.first = nn.Sequential(
            nn.Linear(self.nfeat, self.nhid*4), # 16-48
            nn.BatchNorm1d(self.nhid*4),
            self.nonlinear,
            self.dropout,
        )

        self.lstm = nn.LSTM(input_size=self.nhid * 4,
                            hidden_size=self.nhid,
                            num_layers=1,
                            bidirectional=True)  # 24

        self.second = nn.Sequential(    # 24-12
            nn.Linear(self.nhid * 2, self.nhid),
            self.nonlinear
        )
        self.midLayer = MultiSelfAttention(d_model=self.nhid, heads=nhead, dropout=self.droprate)# (200,1,dropout)


        self.norm = nn.LayerNorm(self.nhid)

        self.linear1 = nn.Linear(self.nhid, self.dim_feedforward)   # (12,24)
        self.linear2 = nn.Linear(self.dim_feedforward, self.nhid)   # (24,12)

        self.SiameseNet = SiameseNet(self.num_features)

    def forward(self, src,batch_node_idx):
        # 1.Linear:
        src = self.bn(src)
        src = self.first(src) # 16-48
        src = src.view(-1, self.path_length, src.size(1)).transpose_(0, 1)   # (batch_size*100,4,48)——(4,batch_size*100,48)

        # 2.LSTM
        src, (ht, ct) = self.lstm(src) # (4,batch_size*100,24)

        # 3.pooling
        _, batch_size, hidlen = src.size()  # batch_size*100,24
        src = src.reshape(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen).mean(dim=2)# 4,batch_size,100,24——>4,batch_szie,24

        # 4. Linear
        src = self.second(src) # 24-12

        # 5.Transformer
        output = self.midLayer(src, src, src)   # (4,batch_szie,12)——>(4,batch_szie,12)
        src = src + self.dropout(output)   # Transformer
        src = self.norm(src)

        # 6.两层forward
        output = self.linear2(self.dropout(self.activation(self.linear1(src))))  # Transformer
        src = src + self.dropout(output)
        output = self.norm(src)  # (4,batch_szie,12)

        # 7.pooling + concat
        selfloop = output[0]
        # output = torch.cat((output.mean(dim=0), selfloop), dim=1)
        output = torch.cat((selfloop, (output[1:].mean(dim=0))), dim=1) # (batch_szie,24)

        score = self.SiameseNet(output)

        idx_1, idx_2 = get_pair_node(batch_node_idx)
        return score ,idx_1,idx_2

"""TransLSTM_A: linear-LSTM-pooling-Linear-Attention-linear-pooling-concat-排序"""
class TransLSTM_A(nn.Module):
    def __init__(self, args, nhead):
        super(TransLSTM_A, self).__init__()
        self.num_paths = args.num_paths
        self.path_length = args.path_length
        self.nclass = args.nclass
        self.nfeat = args.d_e
        self.nhid = args.d_h  #200
        self.droprate = args.dropout

        self.nonlinear = nn.Tanh()
        self.dropout = nn.Dropout(p=self.droprate)   #p: probability of an element to be zeroed. Default: 0.5
        self.activation = F.relu
        self.dim_feedforward=2048

        self.nonlinear = nn.Tanh()
        self.first = nn.Sequential(  # 12047->800
            nn.Linear(self.nfeat, self.nhid*4),
            nn.BatchNorm1d(self.nhid*4),
            self.nonlinear,
            self.dropout,
        )
        self.second = nn.Sequential(    # input = (5,16,400)
            nn.Linear(self.nhid * 2, self.nhid),
            self.nonlinear
        ) # output = (5,16,200)

        self.midLayer = MultiSelfAttention(d_model=self.nhid, heads=nhead, dropout=self.droprate)# (200,1,dropout)

        self.lstm = nn.LSTM(input_size=self.nhid*4,
                            hidden_size=self.nhid,
                            num_layers=1,
                            bidirectional=True)

        self.norm = nn.LayerNorm(self.nhid)

        self.linear1 = nn.Linear(self.nhid, self.dim_feedforward)   # (200,2048)
        self.linear2 = nn.Linear(self.dim_feedforward, self.nhid)   # (2048,200)

        # self.hidden2tag = nn.Sequential(
        #     nn.BatchNorm1d(self.nhid*2), # 400
        #     self.nonlinear,
        #     self.dropout,
        #     nn.Linear(self.nhid*2, self.nclass)  # 9
        # )
        self.ranknet = RankNet(self.num_features)

    def forward(self, src):
        # 1.Linear:
        src = self.first(src) #src = (5,16,100,12047)——>(5,16,100,800)
        src = src.view(-1, self.path_length, src.size(1)).transpose_(0, 1)   # (1600,5,800)——(5,1600,800)

        # 2.LSTM
        src, (ht, ct) = self.lstm(src) # (5,1600,400)双向

        # 3.pooling
        _, batch_size, hidlen = src.size()  # 1600,200
        src = src.reshape(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen).mean(dim=2)# 5,16,100,200——>5,16,200

        # 4. Linear
        src = self.second(src) # (5*16*200)

        # 5.Transformer
        output = self.midLayer(src, src, src)   # (5,16,200)——>(5,16,200)
        # src = src + self.dropout(output)   # Transformer
        # src = self.norm(src)

        # 6.两层forward
        # output = self.linear2(self.dropout(self.activation(self.linear1(src))))  # Transformer
        output = self.linear2(self.dropout(self.activation(self.linear1(output))))
        # src = src + self.dropout(output)
        # output = self.norm(src)  #(5,16,200)

        # 7.pooling + concat
        selfloop = output[0]  # (16,200)
        output = torch.cat((output.mean(dim=0), selfloop), dim=1)  #(16,400)

        # # 8.预测
        # output = self.hidden2tag(output)  #400*9
        # return output

        # 排序模型
        prob = self.ranknet(output)  # batch_size个pairs中i比j大的概率，tensor:(12*11/2,1)，用于计算loss
        scores = self.ranknet.predict(output)  # 对12个节点预测，得到关键度评分(未排序),用于对12个评分排序，计算Diff
        scores_desc = torch.sort(scores, 0, True)[1]  # 降序排序，返回12个节点的降序索引

        return prob, scores_desc

"""TransLSTM_NRA ——>no-RNN ——> linear-pooling-Linear->Attention-linear-pooling-concat-排序"""
class TransLSTM_NRA(nn.Module):
    def __init__(self, args, nhead):
        super(TransLSTM_NRA, self).__init__()
        self.num_paths = args.num_paths
        self.path_length = args.path_length
        self.nclass = args.nclass
        self.nfeat = args.d_e
        self.nhid = args.d_h
        self.droprate = args.dropout
        self.nonlinear = nn.Tanh()
        self.dropout = nn.Dropout(p=self.droprate)  # p: probability of an element to be zeroed. Default: 0.5
        self.activation = F.relu
        self.dim_feedforward = 2048
        self.nonlinear = nn.Tanh()
        self.first = nn.Sequential(
            nn.Linear(self.nfeat, self.nhid *2),
            nn.BatchNorm1d(self.nhid * 2),
            self.nonlinear,
            self.dropout,
        )
        self.second = nn.Sequential(
            nn.Linear(self.nhid * 2, self.nhid),
            self.nonlinear
        )
        self.midLayer = MultiSelfAttention(d_model=self.nhid, heads=nhead, dropout=self.droprate)

        self.norm = nn.LayerNorm(self.nhid)
        self.linear1 = nn.Linear(self.nhid, self.dim_feedforward)
        self.linear2 = nn.Linear(self.dim_feedforward, self.nhid)

        # self.hidden2tag = nn.Sequential(
        #     nn.BatchNorm1d(self.nhid*2),
        #     self.nonlinear,
        #     self.dropout,
        #     nn.Linear(self.nhid * 2, self.nclass)
        # )
        self.ranknet = RankNet(self.num_features)

    def forward(self, src):
        src = self.first(src)  # 输入是16000*12047转换为16000*200
        src = src.view(-1, self.path_length, src.size(1)).transpose_(0, 1)  # 由16000*800转换为 5*3200*200

        _, batch_size, hidlen = src.size()  # _ = 5  batch_size = 3200  hidlen=200
        src = src.reshape(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen).mean(dim=2)

        src = self.second(src)

        output = self.midLayer(src, src, src)   #(5,16,200)
        # src = src + self.dropout(output)  # Transformer
        # src = self.norm(src)

        # output = self.linear2(self.dropout(self.activation(self.linear1(src))))  # Transformer
        output = self.linear2(self.dropout(self.activation(self.linear1(output))))
        # src = src + self.dropout(output)
        # output = self.norm(src)

        selfloop = output[0]
        output = torch.cat((output.mean(dim=0), selfloop), dim=1)  # outputs.mean(dim=0):32*200   outputs:32*400
        # output = self.hidden2tag(output)  # 32*9
        # 排序模型
        prob = self.ranknet(output)  # batch_size个pairs中i比j大的概率，tensor:(12*11/2,1)，用于计算loss
        scores = self.ranknet.predict(output)  # 对12个节点预测，得到关键度评分(未排序),用于对12个评分排序，计算Diff
        scores_desc = torch.sort(scores, 0, True)[1]  # 降序排序，返回12个节点的降序索引

        return prob, scores_desc

"""TransLSTM_NRT ——>no-RNN ——> linear-pooling-Linear->Transformer-linear-pooling-concat-排序"""
class TransLSTM_NRT(nn.Module):
    def __init__(self, args, nhead):
        super(TransLSTM_NRT, self).__init__()
        self.num_paths = args.num_paths
        self.path_length = args.path_length
        self.nfeat = args.d_e  # 6
        self.nhid = args.d_h   # 12
        self.droprate = args.dropout
        self.nonlinear = nn.Tanh()
        self.dropout = nn.Dropout(p=self.droprate)  # p: probability of an element to be zeroed. Default: 0.5
        self.activation = F.relu
        self.dim_feedforward = 6 # 2048
        self.num_features = self.nfeat * 4
        self.nonlinear = nn.Tanh()
        self.first = nn.Sequential(
            nn.Linear(self.nfeat, self.nhid *2),  # 6——24
            nn.BatchNorm1d(self.nhid * 2),
            self.nonlinear,
            self.dropout,
        )
        self.second = nn.Sequential(
            nn.Linear(self.nhid * 2, self.nhid),  # 24——>12
            self.nonlinear
        )
        self.midLayer = MultiSelfAttention(d_model=self.nhid, heads=nhead, dropout=self.droprate)

        self.norm = nn.LayerNorm(self.nhid)   # 12

        self.linear1 = nn.Linear(self.nhid, self.dim_feedforward) # 12——>6
        self.linear2 = nn.Linear(self.dim_feedforward, self.nhid) # 6——>12

        self.ranknet = RankNet(self.num_features)

    def forward(self, src):
        src = self.first(src)  # 输入是16000*12047转换为16000*200，gneE：6->12
        src = src.view(-1, self.path_length, src.size(1)).transpose_(0, 1)  # 由16000*800转换为 5*3200*200

        _, batch_size, hidlen = src.size()  # _ = 5  batch_size = 3200  hidlen=200
        src = src.reshape(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen).mean(dim=2)
        # gneE:src=(5,11,12)
        src = self.second(src)

        output = self.midLayer(src, src, src)   # (5,16,200),gneE:dim = 12
        src = src + self.dropout(output)  # dim = 12
        src = self.norm(src)  # 12

        output = self.linear2(self.dropout(self.activation(self.linear1(src))))  # 12->6->12
        src = src + self.dropout(output)  # 12
        output = self.norm(src)  # 12

        selfloop = output[0]
        output = torch.cat((output.mean(dim=0), selfloop), dim=1)  # outputs.mean(dim=0):32*200   outputs:32*400

        # 排序模型
        prob = self.ranknet(output)  # batch_size个pairs中i比j大的概率，tensor:(12*11/2,1)，用于计算loss
        scores = self.ranknet.predict(output)  # 对12个节点预测，得到关键度评分(未排序),用于对12个评分排序，计算Diff
        scores_desc = torch.sort(scores, 0, True)[1]  # 降序排序，返回12个节点的降序索引

        return prob, scores_desc

"""TransLSTM_NA ——>no-Attention ——> linear-LSTM-Pooling-linear-pooling-concat-排序"""
class TransLSTM_NA(nn.Module):
    def __init__(self, args):
        super(TransLSTM_NA, self).__init__()
        self.num_paths = args.num_paths
        self.path_length = args.path_length
        self.nclass = args.nclass
        self.nfeat = args.d_e
        self.nhid = args.d_h
        self.droprate = args.dropout
        self.nonlinear = nn.Tanh()
        self.dropout = nn.Dropout(p=self.droprate)  # p: probability of an element to be zeroed. Default: 0.5
        self.activation = F.relu
        self.dim_feedforward = 2048
        self.nonlinear = nn.Tanh()
        self.first = nn.Sequential(
            nn.Linear(self.nfeat, self.nhid * 4),
            nn.BatchNorm1d(self.nhid * 4),
            self.nonlinear,
            self.dropout,
        )
        self.second = nn.Sequential(
            nn.Linear(self.nhid * 2, self.nhid),
            self.nonlinear
        )
        self.lstm = nn.LSTM(input_size=self.nhid * 4,
                            hidden_size=self.nhid,
                            num_layers=1,
                            bidirectional=True)

        self.ranknet = RankNet(self.num_features)

    def forward(self, src):
        src = self.first(src)  # 输入是16000*12047转换为16000*200
        src = src.view(-1, self.path_length, src.size(1)).transpose_(0, 1)  # 由16000*800转换为 5*3200*200
        output, (ht, ct) = self.lstm(src)

        _, batch_size, hidlen = output.size()  # _ = 5  batch_size = 3200  hidlen=400
        # outputs= 5*32*100*400.mean压缩了第三个维度
        output = output.view(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen).mean(dim=2)
        output = self.second(output)
        selfloop = output[0]
        output = torch.cat((output.mean(dim=0), selfloop), dim=1)  # outputs.mean(dim=0):32*200   outputs:32*400

        # 排序模型
        prob = self.ranknet(output)  # batch_size个pairs中i比j大的概率，tensor:(12*11/2,1)，用于计算loss
        scores = self.ranknet.predict(output)  # 对12个节点预测，得到关键度评分(未排序),用于对12个评分排序，计算Diff
        scores_desc = torch.sort(scores, 0, True)[1]  # 降序排序，返回12个节点的降序索引

        return prob, scores_desc

"""TransLSTM_NB ——>no RNN、Atention ——> linear-pooling-linear-pooling-concat-排序"""
class TransLSTM_NB(nn.Module):
    def __init__(self, args):
        super(TransLSTM_NB, self).__init__()
        self.num_paths = args.num_paths
        self.path_length = args.path_length
        self.nclass = args.nclass
        self.nfeat = args.d_e
        self.nhid = args.d_h
        self.droprate = args.dropout

        self.nonlinear = nn.Tanh()
        self.dropout = nn.Dropout(p=self.droprate)  # p: probability of an element to be zeroed. Default: 0.5
        self.activation = F.relu
        self.dim_feedforward = 2048

        self.nonlinear = nn.Tanh()
        self.first = nn.Sequential(
            nn.Linear(self.nfeat, self.nhid * 2),
            nn.BatchNorm1d(self.nhid * 2),
            self.nonlinear,
            self.dropout,
        )
        self.second = nn.Sequential(
            nn.Linear(self.nhid * 2, self.nhid),
            self.nonlinear
        )

        self.ranknet = RankNet(self.num_features)

    def forward(self, src):
        src = self.first(src)  # 输入是16000*12047转换为16000*200
        src = src.view(-1, self.path_length, src.size(1)).transpose_(0, 1)  # 由16000*800转换为 5*3200*200

        _, batch_size, hidlen = src.size()  # _ = 5  batch_size = 3200  hidlen=200
        src = src.reshape(self.path_length, int(batch_size / self.num_paths), self.num_paths, hidlen).mean(dim=2)
        src = self.second(src)
        selfloop = src[0]
        output = torch.cat((src.mean(dim=0), selfloop), dim=1)  # outputs.mean(dim=0):32*200   outputs:32*400

        # 排序模型
        prob = self.ranknet(output)  # batch_size个pairs中i比j大的概率，tensor:(12*11/2,1)，用于计算loss
        scores = self.ranknet.predict(output)  # 对12个节点预测，得到关键度评分(未排序),用于对12个评分排序，计算Diff
        scores_desc = torch.sort(scores, 0, True)[1]  # 降序排序，返回12个节点的降序索引

        return prob, scores_desc
