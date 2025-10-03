import torch
import torch.nn as nn

class Actor(nn.Module):
  def __init__(self, state_size, action_size, max_action, fc1_units = 256, fc2_units = 128):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(state_size, fc1_units)
    self.bn1 = nn.BatchNorm1d(fc1_units)
    self.fc2 = nn.Linear(fc1_units, fc2_units)
    self.bn2 = nn.BatchNorm1d(fc2_units)
    self.fc3 = nn.Linear(fc2_units, action_size) 
    self.relu = nn.ReLU()
    self.max_action = max_action

  def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return torch.sigmoid(self.fc3(x)) * self.max_action
  
class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return self.fc3(x)
