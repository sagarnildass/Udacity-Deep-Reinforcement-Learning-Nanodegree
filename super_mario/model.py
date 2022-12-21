import torch
from torch import nn
import copy


class MarioNet(nn.Module):
    '''mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    '''
    def __init__(self, input_size, output_size, seed):
        super().__init__()
        c, h, w = input_size
        self.seed = torch.manual_seed(seed)

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)


class QPixelNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        super(QPixelNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.c1 = nn.Conv3d(in_channels=3, out_channels=10, kernel_size=(1, 5, 5), stride=1)
        self.r1 = nn.ReLU()
        self.max1 = nn.MaxPool3d((1, 2, 2))

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