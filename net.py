import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=32)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(16, 32, kernel_size=32)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv1d(32, 64, kernel_size=16)
        self.conv6 = nn.Conv1d(64, 64, kernel_size=16)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv1d(64, 128, kernel_size=8)
        self.conv8 = nn.Conv1d(128, 128, kernel_size=8)
        self.pool4 = nn.MaxPool1d(kernel_size=5, stride=2)

        self.conv9 = nn.Conv1d(128, 256, kernel_size=3)
        self.conv10 = nn.Conv1d(256, 256, kernel_size=3)
        self.pool5 = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))

        x = self.pool5(x)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, 256)
        x = self.fc(x)
        return x
