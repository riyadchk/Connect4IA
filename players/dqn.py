import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(6 * 7 * 2, 128)  # Assuming a 6x7 board
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 7)  # Output Q-values for each column

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
