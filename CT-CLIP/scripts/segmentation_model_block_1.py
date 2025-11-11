import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom single deconvolution block that accepts custom parameters.
class CustomSingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(
            in_planes, out_planes, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            output_padding=output_padding
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.deconv(x))

# Simple convolution block used after the deconvolution.
class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        # Use "same" padding: padding = kernel_size//2 for each dimension if kernel_size is an int;
        # if kernel_size is a tuple, compute element-wise.
        if isinstance(kernel_size, int):
            pad = kernel_size // 2
            padding = (pad, pad, pad)
        else:
            padding = tuple(k // 2 for k in kernel_size)
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding)
    
    def forward(self, x):
        return self.conv(x)

# Provided Deconv3DBlock using our custom deconv block.
class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.block = nn.Sequential(
            CustomSingleDeconv3DBlock(in_planes, out_planes, kernel_size, stride, padding, output_padding),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

# Upsample network that applies three blocks sequentially.
class UpsampleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: From (512, 24, 24, 24) to (256, 48, 48, 24)
        self.block1 = Deconv3DBlock(
            in_planes=512, 
            out_planes=256, 
            kernel_size=(2,2,1), 
            stride=(2,2,1), 
            padding=(0,0,0), 
            output_padding=(0,0,0)
        )
        # Block 2: From (256, 48, 48, 24) to (64, 120, 120, 60)
        self.block2 = Deconv3DBlock(
            in_planes=256, 
            out_planes=64,
            kernel_size=(3,3,3),
            stride=(3,3,3),
            padding=(12,12,6),      # chosen so that: (48-1)*3 - 2*12 + 3 = 120 for dims 1 & 2; and (24-1)*3 - 2*6 + 3 = 60 for dim 3.
            output_padding=(0,0,0)
        )
        # Block 3: From (64, 120, 120, 60) to (16, 240, 240, 120)
        self.block3 = Deconv3DBlock(
            in_planes=64, 
            out_planes=16,
            kernel_size=(2,2,2),
            stride=(2,2,2),
            padding=(0,0,0),
            output_padding=(0,0,0)   # (120-1)*2+2 = 240 and (60-1)*2+2 = 120.
        )
    
    def forward(self, x):
        # x is expected to be of shape (B, 512, 24, 24, 24)
        x = self.block1(x)  # Expected shape: (B, 256, 48, 48, 24)
        x = self.block2(x)  # Expected shape: (B, 64, 120, 120, 60)
        x = self.block3(x)  # Expected shape: (B, 16, 240, 240, 120)
        return x

# Test the network:
if __name__ == "__main__":
    net = UpsampleNetwork()
    x = torch.randn(1, 512, 24, 24, 24)
    y = net(x)
    print("Output shape:", y.shape)  # Expected: torch.Size([1, 16, 240, 240, 120])
