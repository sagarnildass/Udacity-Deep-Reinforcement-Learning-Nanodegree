import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import flatten_conv_feature


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        self.action_size = action_size
        self.value_function_fc = nn.Linear(fc2_units, 1)
        self.advantage_function_fc = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        value_function = self.value_function_fc(x)
        advantage_function = self.advantage_function_fc(x)
        
        return value_function + advantage_function - advantage_function.mean(1).unsqueeze(1).expand(x.size(0), self.action_size) / self.action_size


class QPixelNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, training):
        super(QPixelNetwork, self).__init__()

        self.c1 = nn.Conv3d(in_channels=3, out_channels=10, kernel_size=(1, 5, 5), stride=1)
        self.r1 = nn.ReLU()
        self.max1 = nn.MaxPool3d((1, 2, 2))
        # [(W−K+2P)/S]+1
        # (32-5+ 0)/1 + 1 -> 28x28x10 -> 14x14x10
        # (28-5 +0)+1 -> 24x24x10 -> 12x12x10
        self.c2 = nn.Conv3d(in_channels=10, out_channels=32, kernel_size=(1, 5, 5), stride=1)
        self.r2 = nn.ReLU()
        self.max2 = nn.MaxPool3d((1, 2, 2))

        # 14-5 +1 -> 5x5x32
        # 12-5 + 1 -> 4x4x32
        self.fc4 = nn.Linear(4 * 4 * 32 * 3, action_size)

    #         self.r4 = nn.ReLU()
    #         self.fc5 = nn.Linear(84, action_size)

    def forward(self, img_stack):
        #         print('-',img_stack.size())
        output = self.c1(img_stack)

        output = self.r1(output)
        output = self.max1(output)
        #         print('*',output.size())

        output = self.c2(output)
        output = self.r2(output)
        output = self.max2(output)
        #         print('**',output.size())

        output = output.view(output.size(0), -1)
        #         print('***', output.size())
        output = self.fc4(output)
        return output