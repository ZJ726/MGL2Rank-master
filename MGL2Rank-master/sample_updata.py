coding='utf-8'
import numpy.random as npr
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,vstack
import time
from new_dataloader import *
from sklearn.preprocessing import normalize

class walk_dic_featwalk_new:
    def __init__(self, Net, featur, num_paths, path_length, alpha):
        self.alpha = alpha
        self.num_paths = num_paths
        self.path_length = path_length
        self.n, self.m = featur.shape
        self.Net = normalize(csr_matrix(Net), norm='l1', axis=0)
        self.featur = normalize(csr_matrix(featur), norm='l1',axis=0)

        self.path_list_Net = []   # 路径存在列表中
        self.qListrow = []  # for each instance
        self.JListrow = []
        self.idx = []

        # 1.节点---->节点 和 节点---->属性
        net_featur = normalize(vstack([self.alpha * self.Net.T, (1 - self.alpha) * self.featur.T]), norm='l1', axis=0)
        '''
            [    alpha*G
              (1-alpha)*AT  ]
        '''
        net_featur.eliminate_zeros()
        for ni in range(self.n):
            self.coli = net_featur.getcol(ni)
            data = self.coli
            self.path_list_Net.append(self.coli.nnz)
            J,q = alias_setup(self.coli.data)   # 每一列数据作为 probs,别名采样,得到Alias和probs数组。

            self.JListrow.append(J)  # int, Alias
            self.qListrow.append(q)  # float, probs
            self.idx.append(self.coli.indices)  # 非0元素的行索引idx

    def function(self):
        print("Start sample!")
        start_time = time.time()
        sentencedic = [[] for _ in range(self.n)]
        # sentnumdic = [[] for _ in range(self.n)]

        allidx = np.nonzero(self.path_list_Net)[0]  #待嵌入的节点

        if len(allidx) != self.n:
            for i in np.where(np.asarray(self.path_list_Net) == 0)[0]:
                sentencedic[i] = [i] * (self.path_length * self.num_paths)
        features1 = self.featur
        features1.eliminate_zeros()

        for i in allidx:
            sentence = []
            for j in range(self.num_paths):
                sentence.append(i)
                current = i
                for senidx in range(self.path_length - 1):
                    if current >= self.n : # 说明从f出发的采样过程
                        reference = features1[sentence[-2],current - self.n]
                        # features1 = features1.todense()
                        currf = features1[:,current - self.n].toarray()
                        diff = np.where(currf == 0., 1., currf - reference)
                        probs = normalize((1 - abs(diff).reshape(-1, 1)), norm='l1', axis=0)

                        J, q = alias_setup(probs)
                        self.JListrow.append(J)
                        self.qListrow.append(q)
                        dataJ = self.JListrow
                        dataq = self.qListrow

                        self.accessnode = features1.getcol(current - self.n) # 不含0值
                        testaccessnode = self.accessnode
                        self.idx.append(self.accessnode.indices)  # 行索引idx(0-5)
                        data = self.idx

                        fromcurrnode_choose1node = alias_draw(dataJ[self.n], dataq[self.n])  # 第5个数
                        # accessibility_f = self.idx[self.n]
                        # current = accessibility_f[fromcurrnode_choose1node]
                        current = fromcurrnode_choose1node
                        sentence.append(current)

                        self.JListrow.pop()
                        self.qListrow.pop()
                        self.idx.pop()

                    else:
                        fromcurrnode_choose1node  = alias_draw(self.JListrow[current], self.qListrow[current])
                        accessibility = self.idx[current]
                        current =  accessibility[fromcurrnode_choose1node]
                        sentence.append(current)


            sentencedic[i] = sentence
        end_time = time.time()
        print("sample finished: '{}'".format(sentencedic))
        print('sample took %f second' % (end_time - start_time))
        return sentencedic#,sentnumdic

def alias_setup(probs):  # 输入转移概率probs
    '''
    probs:某个概率分布
    返回: Alias数组与Prob数组
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=int)
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

# 通过拼凑，将各个类别都凑为1
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop() # 某个凹陷列的索引值
        large = larger.pop()  # 某个突出列的索引值

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q

def alias_draw(J, q):
    K = len(J)
    kk = int(np.floor(npr.rand() * K))
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]
