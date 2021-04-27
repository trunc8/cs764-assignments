# Imports
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn

import torch.optim as optim

with open('data_train_lift.pkl', 'rb') as f:
    data = pickle.load(f)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

test_X = torch.Tensor(data['joint_2d_1']).view(-1,15*2)
test_y = torch.Tensor(data['joint_3d']).view(-1,15*3)


class LiftModel(nn.Module):
    '''
    The model architecture
    '''
    def __init__(self, n_blocks=2, hidden_layer=1024, dropout=0.1, output_nodes=15*3):
        super().__init__()
        
        self.n_blocks = n_blocks
        self.hidden_layer = hidden_layer
        self.dropout = dropout
        self.output_nodes = output_nodes
        
        self.fc1 = nn.Linear(15*2, self.hidden_layer)

        self.res = nn.Sequential(
            nn.Linear(self.hidden_layer, self.hidden_layer),
            nn.BatchNorm1d(self.hidden_layer),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_layer, self.hidden_layer),
            nn.BatchNorm1d(self.hidden_layer),
            nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )
        
        self.fc2 = nn.Linear(self.hidden_layer, self.output_nodes)
    
    def forward(self, x):
        x = self.fc1(x)
        for _ in range(self.n_blocks):
            y = self.res(x)
            x = x+y
        y = self.fc2(x)
        return y


def cal_mpjpe(P1, P2, avg=True):
    pose_1 = P1.view(-1, 15, 3)
    pose_2 = P2.view(-1, 15, 3)
    diff = pose_1 - pose_2
    diff_sq = torch.square(diff)
    dist_per_joint = torch.sqrt(torch.sum(diff_sq, dim=2))
    dist_per_sample = torch.mean(dist_per_joint, dim=1)
    if avg is True:
        dist_avg = torch.mean(dist_per_sample)
    else:
        dist_avg = dist_per_sample
    return dist_avg


def test(net):
    tot_error = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            pose_2 = test_y[i].to(device)
            pose_1 = net(test_X[i].view(-1, 15*2).to(device))[0]  # returns a list
            tot_error += cal_mpjpe(pose_1, pose_2)
            total += 1
    print("Average error: ", round(float(tot_error)/total, 3))



PATH = "liftModel"
net = LiftModel().to(device)
net.load_state_dict(torch.load(PATH))
net.eval()
test(net)


