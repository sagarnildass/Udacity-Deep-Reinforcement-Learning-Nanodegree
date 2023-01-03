import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """QNetwork.

    Simple Dense neural network
    to serve as funcction approximator.
    """

    def __init__(
            self,
            action_size,
            seed,
            in_channels=3,
            conv1_kernel=3,
            conv1_filters=16,
            conv1_strides=1,
            conv2_kernel=3,
            conv2_filters=32,
            conv2_strides=1,
            fc1_units=512,
            fc2_units=512
    ):
        super(QNetwork, self).__init__()
        self.seed = seed
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, conv1_filters, kernel_size=conv1_kernel, stride=conv1_strides, padding=1),
            nn.BatchNorm2d(conv1_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=conv2_kernel, stride=conv2_strides, padding=1),
            nn.BatchNorm2d(conv2_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(conv2_filters * 21 * 21, fc1_units),
            nn.BatchNorm1d(fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.BatchNorm1d(fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )

    def forward(self, x):
        # print(self.network(x).shape)
        return self.network(x)