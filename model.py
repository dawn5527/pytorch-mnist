import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim


class LeNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)#输入和输出通道数分别为1和10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)#输入和输出通道数分别为10和20
        self.conv2_drop = nn.Dropout2d()#随机选择输入的信道，将其设为0
        self.fc1 = nn.Linear(16*4*4, 120)#输入的向量大小和输出的大小分别为320和50
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))#conv->max_pool->relu
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))#conv->dropout->max_pool->relu
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))#fc->relu
        x = F.dropout(x, training=self.training)#dropout
        x = F.relu(self.fc2(x))  # fc->relu
        x = self.fc3(x)
        return F.log_softmax(x)
