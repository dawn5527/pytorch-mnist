import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        #1*28*28->6*24*24
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        #6*12*12->16*8*8
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        #dropout
        self.conv2_drop = nn.Dropout2d()
        #6*4*4->120
        self.fc1 = nn.Linear(16*4*4, 120)
        #120->84
        self.fc2 = nn.Linear(120, 84)
        #84->10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #conv->max_pool->relu
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #conv->dropout->max_pool->relu
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 16*4*4)
        #fc->relu
        x = F.relu(self.fc1(x))
        #dropout
        x = F.dropout(x, training=self.training)
        # fc->relu
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)
    
    def predict(self, x):
        #conv->max_pool->relu
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #conv->dropout->max_pool->relu
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 16*4*4)
        #fc->relu
        x = F.relu(self.fc1(x))
        #dropout
        x = F.dropout(x, training=self.training)
        # fc->relu
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)
        return F.softmax(x,dim=1)