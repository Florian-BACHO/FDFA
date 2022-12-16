import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.DFATrainingHook import DFATrainingHook
from utils.OutputTrainingHook import OutputTrainingHook

class Network(nn.Module):
    def __init__(self, batch_size, train_mode, device):
        super(Network, self).__init__()
        self.batch_size = batch_size
        self.train_mode = train_mode
        self.device = device

        self.conv1 = nn.Conv2d(1, 15, 5, 1)
        self.conv1_bn = nn.BatchNorm2d(15)
        self.conv1_dfa = DFATrainingHook(train_mode, device)

        self.conv2 = nn.Conv2d(15, 40, 5, 1)
        self.conv2_bn = nn.BatchNorm2d(40)
        self.conv2_dfa = DFATrainingHook(train_mode, device)

        self.fc1 = nn.Linear(4000, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc1_dfa = DFATrainingHook(train_mode, device)

        self.fc_out = nn.Linear(128, 10)

        self.output_hook = OutputTrainingHook(train_mode)


    def forward(self, x):
        grad_at_output = torch.zeros([x.shape[0], 10], requires_grad=False).to(self.device)
        dir_der_at_output = torch.zeros([x.shape[0], 10], requires_grad=False).to(self.device)
        
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.conv1_dfa(x, dir_der_at_output, grad_at_output)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.conv2_dfa(x, dir_der_at_output, grad_at_output)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc1_dfa(x, dir_der_at_output, grad_at_output)

        x = self.fc_out(x)

        x = self.output_hook(x, dir_der_at_output, grad_at_output)

        return x