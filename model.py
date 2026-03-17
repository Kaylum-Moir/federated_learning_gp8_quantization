import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3)
        self.fc1 = nn.Linear(5408, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        return x