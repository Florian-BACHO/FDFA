import torch
import torch.nn as nn

from utils.DFATrainingHook import DFATrainingHook
from utils.OutputTrainingHook import OutputTrainingHook


class Network(nn.Module):
    def __init__(self, batch_size, n_hidden, train_mode, device):
        super(Network, self).__init__()
        self.batch_size = batch_size
        self.train_mode = train_mode
        self.device = device

        last_input_size = 784
        fc_hidden = []
        for _ in range(n_hidden):
            fc_hidden.append(nn.Linear(last_input_size, 800).to(device))
            last_input_size = 800
        self.fc_hidden = nn.ModuleList(fc_hidden)

        self.fc_out = nn.Linear(800, 10).to(device)


    def forward(self, x):
        x = torch.flatten(x, 1)

        for fc in self.fc_hidden:
            x = fc(x)
            x = nn.functional.relu(x)

        x = self.fc_out(x)

        return x