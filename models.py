import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArtifactDetectionNN(nn.Module):
    def __init__(self, input_dim):
        super(ArtifactDetectionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ArtifactDetectionCNN(nn.Module):
    def __init__(self, input_dim):
        super(ArtifactDetectionCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.flatten_dim = self._get_flatten_dim(input_dim)

        self.fc1 = nn.Linear(self.flatten_dim, 256)  # Dense(256) in Keras
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)  # Dense(128) in Keras
        self.fc3 = nn.Linear(128, 1)  # Dense(1) in Keras (no activation for regression)

    def _get_flatten_dim(self, input_dim):
        x = torch.zeros(1, 1, input_dim)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        flatten_dim = x.numel()
        return flatten_dim

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x