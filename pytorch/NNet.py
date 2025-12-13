import torch
import torch.nn as nn
import torch.nn.functional as F


class FreecivNNet(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        board_shape = game.getBoardSize()
        self.board_c, self.board_h, self.board_w = board_shape
        self.action_size = game.getActionSize()
        # Optional head sizes (move/attack/econ) for multi-head policy output.
        self.policy_head_sizes = getattr(game, "policy_head_sizes", None)
        channels = args.num_channels

        self.conv1 = nn.Conv2d(self.board_c, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)

        flat_size = channels * self.board_h * self.board_w
        if self.policy_head_sizes:
            assert sum(self.policy_head_sizes) == self.action_size, "Head sizes must sum to action_size"
            self.policy_move = nn.Linear(flat_size, self.policy_head_sizes[0])
            self.policy_attack = nn.Linear(flat_size, self.policy_head_sizes[1])
            self.policy_econ = nn.Linear(flat_size, self.policy_head_sizes[2])
        else:
            self.policy_head = nn.Linear(flat_size, self.action_size)
        self.value_head = nn.Sequential(
            nn.Linear(flat_size, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )

    def forward(self, x):
        # x shape: (batch, C, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        if self.policy_head_sizes:
            logits = [
                self.policy_move(x),
                self.policy_attack(x),
                self.policy_econ(x),
            ]
            policy = F.log_softmax(torch.cat(logits, dim=1), dim=1)
        else:
            policy = F.log_softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value
