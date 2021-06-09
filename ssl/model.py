import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lin = nn.Linear(32 * 18 * 18, 256)  # 40
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.lin(x))
        out = self.out(x)

        return out
