import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # State value 
        self.state_value0 = nn.Linear(state_size, 128)
        self.state_value1 = nn.Linear(128, 64)
        self.state_value2 = nn.Linear(64, 1)
        # Advantage value 
        self.advantage_value0 = nn.Linear(state_size, 128)
        self.advantage_value1 = nn.Linear(128, 64)
        self.advantage_value2 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        # State value branch
        x_state = F.relu(self.state_value0(state))
        x_state = F.relu(self.state_value1(x_state))
        x_state = F.relu(self.state_value2(x_state))
        
        # Advantage value branch
        x_advantage = F.relu(self.advantage_value0(state))
        x_advantage = F.relu(self.advantage_value1(x_advantage))
        x_advantage = F.relu(self.advantage_value2(x_advantage))
        
        return x_advantage + x_state
        