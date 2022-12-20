import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


##Hidden layer's weights intialization
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        _input_layer = 128
        _hidden_1 = 256
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, _input_layer)
        self.fc2 = nn.Linear(_input_layer, _hidden_1)
        self.fc3 = nn.Linear(_hidden_1, action_size)

        # batchnorm
        self.bn_input = nn.BatchNorm1d(_input_layer)
        self.bn_hidden = nn.BatchNorm1d(_hidden_1)

        # Dropout
        self.dpout = nn.Dropout(p=0.20)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # If there are multiple windows, the state will have them already concatenated
        # Input layer: Dense + Batchnorm + ReLU
        x = self.fc1(state)
        x = self.bn_input(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        out = torch.tanh(x)
        return out


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        _input_layer = 128
        _hidden_1 = 256
        self.seed = torch.manual_seed(seed)

        self.fcs1 = nn.Linear(state_size, _input_layer)
        self.fc2 = nn.Linear(_input_layer + action_size, _hidden_1)  # concat the action from the actor
        self.fc3 = nn.Linear(_hidden_1, 1)

        # batchnorm
        self.bn_input = nn.BatchNorm1d(_input_layer)

        # Dropout
        self.dpout = nn.Dropout(p=0.20)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Input layer: Dense + Batchnorm + ReLU
        x = self.fcs1(state)
        x = self.bn_input(x)
        x = F.relu(x)

        # First hidden layer Dense + ReLU
        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        x = F.relu(x)

        # Output layer: Dense
        out = self.fc3(x)
        return out
