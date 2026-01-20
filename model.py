import torch
import torch.nn as nn 
import torch.nn.functional as F 
import config 

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, padding=1):
        super().__init__()
        outchannels = out_channels or config.NUM_CHANNELS
        self.conv = nn.Conv2d(in_channels,out_channels, kernel_size=kernel_size, padding = padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x 


class ResBlock(nn.Module):
    def __init__(self,channels = None):
        super().__init__()
        channels = channels or config.NUM_CHANNELS
        self.conv1 = nn.Conv2d(channels,channels,3,padding=1,bias = False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels,channels,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(channels)
    def forward(self,x):
        residual = x 
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out+=residual 
        return F.relu(out)

class PolicyHead(nn.Module):
    def __init__(self,in_channels = None, action_size = None):
        super().__init__()
        in_channels = in_channels or config.NUM_CHANNELS
        action_size = action_size or config.ACTION_SIZE

        self.conv = nn.Conv2d(in_channels, config.POLICY_HEAD_CHANNELS, kernel_size=1)
        self.bn = nn.BatchNorm2d(config.POLICY_HEAD_CHANNELS)
        self.fc = nn.Linear(config.POLICY_HEAD_CHANNELS*8*8, action_size)
    
    def forward(self,x,legal_mask = None):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        if legal_mask is not None:
            x = x.masked_fill(legal_mask==0, float('-inf'))
        x = F.softmax(x, dim=1)
        return x 

class ValueHead(nn.Module):
    def __init__(self,in_channels = None):
        super().__init__()
        in_channels = in_channels or config.NUM_CHANNELS

        self.conv = nn.Conv2d(in_channels, config.VALUE_HEAD_CHANNELS, kernel_size=1)
        self.bn = nn.BatchNorm2d(config.VALUE_HEAD_CHANNELS)
        self.fc1 = nn.Linear(config.VALUE_HEAD_CHANNELS*8*8, 256)
        self.fc2 = nn.Linear(256,1)
    def forward(self,x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

class ChessModel(nn.Module):
    def __init__(self, in_channels = 21, num_res_blocks = None, action_size = None):
        super().__init__()
        num_res_blocks = num_res_blocks or config.NUM_RES_BLOCKS
        action_size = action_size or config.ACTION_SIZE

        self.conv_block = ConvBlock(in_channels, config.NUM_CHANNELS)

        self.res_blocks = nn.ModuleList(
            [ResBlock(config.NUM_CHANNELS) for _ in range(num_res_blocks)]
        )

        self.policy_head = PolicyHead(config.NUM_CHANNELS, action_size)
        self.value_head = ValueHead(config.NUM_CHANNELS)
    
    def forward(self,x, legal_mask = None):
        x = self.conv_block(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        
        policy = self.policy_head(x, legal_mask)
        value = self.value_head(x)

        return policy, value

