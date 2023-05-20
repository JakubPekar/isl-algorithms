import torch
import torch.nn.functional as F
from torch import nn


class E2E_CNN(nn.Module):
    def __init__(self, in_channels: int, channel_size: int):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 96, 7)
        self.conv2 = nn.Conv1d(96, 96, 7)
        self.conv3 = nn.Conv1d(96, 128, 5)
        self.conv4 = nn.Conv1d(128, 128, 5)
        self.conv5 = nn.Conv1d(128, 128, 3)

        self.pool1 = nn.MaxPool1d(7)
        self.pool2 = nn.MaxPool1d(5)
        self.pool3 = nn.MaxPool1d(5)

        n = (((channel_size - 6) // 7 - 10) // 5 - 4) // 5 - 2

        self.fc1 = nn.Linear(n * 128, 500)
        self.fc2 = nn.Linear(500, 3)

        self.drop = nn.Dropout1d(0.5)



    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool3(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)
