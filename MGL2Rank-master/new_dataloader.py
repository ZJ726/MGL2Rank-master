import random
import time
import networkx as nx
import torch
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import scipy.io as sio
import math
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

def dataloader_test_929new_npart():
    np.random.seed(3407)
    # 1.准备A与G
    G = pd.read_csv(r'D:\myfile\mypaper\paper0\exp2\MGLRank-master\data\ShenYangdata\929_new\929_G.csv', header=None)
    adj = torch.tensor(G.values)

    A = pd.read_csv(r'D:\myfile\mypaper\paper0\exp2\MGLRank-master\data\ShenYangdata\929_new\929_16_A.csv')
    features = torch.tensor(A.values)
    n, m = features.shape  # 1002,6

    score_mis_order = pd.read_csv(r'D:\myfile\mypaper\paper0\exp2\MGLRank-master\data\ShenYangdata\929_new\score_929.csv',header=None)
    score_mis_order = torch.tensor(score_mis_order.values, dtype=torch.float64)

    nodes = np.arange(n)
    # 定义每个节点的类别
    node_classes = np.zeros(n)
    node_classes[0:200] = 0
    node_classes[200:400] = 1
    node_classes[400:600] = 2
    node_classes[600:800] = 3
    node_classes[800:929] = 4

    train_val_nodes = []
    test_nodes = []

    skf_outer = StratifiedKFold(n_splits=7, shuffle=True, random_state=3407) # 7
    for train_val_indices, test_indices in skf_outer.split(nodes, node_classes):
        train_val_nodes = nodes[train_val_indices]
        test_nodes = nodes[test_indices]
    np.random.shuffle(test_nodes)

    # return train_val_nodes,test_nodes
    node_class = [0] * 133 + [1] * 133 + [2] * 133 + [3] * 133 + [4] * 133 + [5] * 132  # train_val_nodes中的类别比例

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.165, random_state=3407) # 0.17
    train_idx, val_idx = next(sss.split(train_val_nodes, node_class))  # 生成划分索引
    train_nodes = [train_val_nodes[i] for i in train_idx]  # 根据划分索引获取训练集节点
    val_nodes = [train_val_nodes[i] for i in val_idx]  # 根据划分索引获取验证集节点

    GT_rank = pd.read_csv(r'D:\myfile\mypaper\paper0\exp2\MGLRank-master\data\ShenYangdata\929_new\GT_rank.csv')
    node_ranking = GT_rank['id_des']
    node_ranking = node_ranking.values.tolist()
    node_ranking = np.array(node_ranking)

    idx_train = node_ranking[train_nodes]
    idx_val = node_ranking[val_nodes]
    idx_test = node_ranking[test_nodes]

    idx_train = torch.tensor(idx_train, dtype=torch.long)  # 一个tensor，内部是一个训练集节点索引列表
    idx_val = torch.tensor(idx_val, dtype=torch.long)
    idx_test = torch.tensor(idx_test, dtype=torch.long)

    start_time = time.time()
    return adj, features, score_mis_order, idx_train,idx_val,idx_test,start_time # # # #0.74 #有向图

def get_groundtruth_BCE(batch_node_idx, score_mis_order):
    pairs = []
    for i, node_i_idx in enumerate(batch_node_idx):  # node_i_idx=4466
        for node_j_idx in batch_node_idx[i + 1:]:  # j=0,5971,1325,5849
            pair = torch.tensor([node_i_idx, node_j_idx])
            pairs.append(pair)
    pairs_idx = torch.stack(pairs)

    pairs_GT = []
    for i in range(0, pairs_idx.size(0)):  # 0~9
        pairs_gt = score_mis_order[pairs_idx[i]]
        pairs_GT.append(pairs_gt)
    pairs_score = torch.stack(pairs_GT)

    pij = []
    for i in range(0, pairs_score.size(0)):  # i=0~54
        if pairs_score[i][0] > pairs_score[i][1]:
            scalar = 1
        else:
            scalar = 0
        pij.append(scalar)
    pij = torch.Tensor(pij)
    shape = pij.shape[0]
    pij = pij.reshape(shape, 1)
    return pij

def getPredOrder(idx_1,idx_2,pred_res,batch_node_idx):
    batch_node_idx = np.array(batch_node_idx)
    list_ = list(batch_node_idx)
    adj_matrix = torch.zeros((len(list_), len(list_)))
    for i in range(len(idx_1)):
        adj_matrix[list_.index(idx_1[i]), list_.index(idx_2[i])] = pred_res[i]
        adj_matrix[list_.index(idx_2[i]), list_.index(idx_1[i])] = 1 - pred_res[i]

    sorted_indices = torch.argsort(adj_matrix.sum(dim=1), descending=True)

    rank = []
    for i in sorted_indices:
        rank.append(list_[i.item()])
    return rank

def getLabelOrder(batch_node_idx):
    score = pd.read_csv("D:\myfile\mypaper\paper0\exp2\MGLRank-master\data\ShenYangdata\929_new\score_929.csv",header=None)
    score = torch.tensor(score.values)
    batch_score = score[batch_node_idx] # batch_node_idx的对应分数
    sorted_idx = torch.argsort(batch_score.squeeze(), descending=True) # batch_node_idx的降序分数
    label_order = batch_node_idx[sorted_idx] # batch_node_idx的对应分数
    return label_order

def measure(preds_order, GT_order,batch_size):
    diff = []
    for i, node_id_i in enumerate(preds_order):  # 0:2,  1:4,   2:1,   3:3,   4:5,   5:0
        for j, node_id_j in enumerate(GT_order):  # 0:3,  1:4,   2:1,   3:0,   4:2,   5:5
            if node_id_i == node_id_j:
                sub = abs(i - j)
                diff.append(sub)
    diff = np.sum(diff)
    diff = diff / math.floor(batch_size ** 2 / 2)  # 一个batch的diff,args.batch_size
    return diff

def getDiGraph():
    tree = ET.parse('D:\App\sumofile\map_Shenyang\groundtruth\929_new\edge_conn.xml')
    root = tree.getroot()
    Edge = []
    Conn = []
    G = nx.DiGraph()
    # 添加节点
    for edge in root.findall('edge'):
        Edge.append(edge.get('id')) # ['-119086807#0', '-119086807#1',
    for conn in root.findall('connection'):
        fromto = conn.get('from'), conn.get('to')
        Conn.append(fromto)         # [('-119086807#0', '119086807#0'), ('-119086807#1', '-119086807#0'),
    node_list = Edge
    edge_list = Conn
    id2int = {}
    for i, node_id in enumerate(node_list):
        id2int[node_id] = i
    # 将边列表中的节点ID替换为整数值
    new_edge_list = []
    for edge in edge_list:
        u, v = edge
        new_edge_list.append((id2int[u], id2int[v]))

    G.add_nodes_from(list(id2int.values()))
    G.add_edges_from(new_edge_list)
    return G

