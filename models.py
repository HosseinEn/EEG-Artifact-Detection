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


class ExtremeLearningMachine(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3):
        super(ExtremeLearningMachine, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_weights = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim), requires_grad=False)
        self.bias = nn.Parameter(torch.randn(self.hidden_dim), requires_grad=False)
        self.output_weights = nn.Parameter(torch.randn(hidden_dim, output_dim))

    def forward(self, x):
        h = F.sigmoid(torch.mm(x, self.input_weights) + self.bias)
        y = torch.mm(h, self.output_weights)
        return y

    def train_elm(self, x_train, y_train):
        y_train = F.one_hot(y_train.long(), num_classes=self.output_dim)
        H = F.sigmoid(torch.matmul(x_train, self.input_weights) + self.bias)
        H_pinv = torch.pinverse(H)
        output_weights_new = torch.matmul(H_pinv, y_train.float())
        with torch.no_grad():
            self.output_weights.copy_(output_weights_new)