import torch
import torch.nn as nn
import torch.nn.functional as F


class FreecivNNet(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        board_shape = game.getBoardSize()
        self.board_c, self.board_h, self.board_w = board_shape
        self.action_size = game.getActionSize()
        channels = args.num_channels

        self.conv1 = nn.Conv2d(self.board_c, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)

        self.policy_head = nn.Linear(channels * self.board_h * self.board_w, self.action_size)
        self.value_head = nn.Sequential(
            nn.Linear(channels * self.board_h * self.board_w, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )

    def forward(self, x):
        # x shape: (batch, C, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        policy = F.log_softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value
