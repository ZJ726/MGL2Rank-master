import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline.rank.RankNet.ranknet_model import get_pair_node

class SiameseNet(nn.Module):
    def __init__(self, num_features):
        super(SiameseNet, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)  # 64
        self.fc2 = nn.Linear(64, 24)            # 64,24
        self.fc3 = nn.Linear(24, 8)             # 24 8
        self.fc4 = nn.Linear(16, 1)              # 16, 1
        self.output = nn.Sigmoid()

    def forward_once(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) # x = self.bn3(F.relu(self.fc3(x)))
        return x

    def forward(self,output):
        input_1, input_2 = get_pair_node(output)
        if input_1 == None or input_2 == None:
            return None

        out1 = self.forward_once(input_1)
        out2 = self.forward_once(input_2)

        dist = torch.cat([out1,out2],1)
        dist = self.fc4(dist)
        score = self.output(dist)

        return score

def triplet_loss(score, label):
    margin = 1.0
    # loss = torch.mean(torch.max(torch.zeros_like(score), margin - label * score))
    loss = torch.where(label == 0, torch.zeros_like(score), torch.max(torch.zeros_like(score), margin - label * score))
    loss = torch.mean(loss)
    return loss


