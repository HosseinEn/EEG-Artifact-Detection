import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ArtifactDetectionNN(nn.Module):
    def __init__(self, input_dim):
        super(ArtifactDetectionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')  # For ReLU
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')

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

        self.flatten_dim = self._get_flatten_dim(input_dim)
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, dropout=0.5)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, dropout=0.5)

        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)

    def _get_flatten_dim(self, input_dim):
        x = torch.zeros(1, 1, input_dim)
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))
        x = self.pool3(self.bn3(self.conv3(x)))
        return x.shape[-1]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.permute(0, 2, 1)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = x[:, -1, :]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#####################################3 RNN ########################################

class ArtifactDetectionRNN(nn.Module):
    def __init__(self, input_dim):
        super(ArtifactDetectionRNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.flatten_dim = self._get_flatten_dim(input_dim)

        self.rnn = nn.RNN(input_size=128, hidden_size=256, batch_first=True, nonlinearity='tanh')
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, dropout=0.5)

        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

    def _get_flatten_dim(self, input_dim):
        x = torch.zeros(1, 1, input_dim)
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))
        x = self.pool3(self.bn3(self.conv3(x)))
        return x.shape[-1]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.permute(0, 2, 1)
        x, h_n_rnn = self.rnn(x)
        x, (h_n_lstm, c_n_lstm) = self.lstm(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

