import torch
import torch.nn as nn
import torch.nn.functional as F
from pixel_utils import flatten_conv_feature

class QPixelNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QPixelNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(10*10*64, 512)
        self.fc2 = nn.Linear(512, self.action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.bn1(F.relu(self.conv1(state)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = x.view(-1, flatten_conv_feature(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x