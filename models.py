import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArtifactDetectionNN(nn.Module):
    def __init__(self, input_dim):
        super(ArtifactDetectionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # 3 classes: 0, 1, 2
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
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Calculate the output size after convolution and pooling layers
        self.flatten_dim = self._get_flatten_dim(input_dim)

        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)


    def _get_flatten_dim(self, input_dim):
        x = torch.zeros(1, 1, input_dim)  # Batch size 1, channels 1, length input_dim
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))
        x = self.pool3(self.bn3(self.conv3(x)))
        flatten_dim = x.numel()
        return flatten_dim

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

