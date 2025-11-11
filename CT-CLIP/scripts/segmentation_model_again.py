import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.functional as nnf

class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UNETR(nn.Module):
    def __init__(self, input_dim=1, output_dim=2, embed_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.ft = 16

        self.downsample_z0 = Downsample3D(in_channels=1, out_channels=1)
        self.downsample = nn.ModuleList([Downsample3D(in_channels=512, out_channels=512) for _ in range(4)])

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(input_dim, self.ft, 3),
                Conv3DBlock(self.ft, self.ft*2 , 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, self.ft * 16),
                Deconv3DBlock(self.ft * 16, self.ft * 8),
                Deconv3DBlock(self.ft * 8, self.ft * 4)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, self.ft * 16),
                Deconv3DBlock(self.ft * 16, self.ft * 8),
            )

        self.decoder9 = \
            Deconv3DBlock(embed_dim, self.ft * 16)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(embed_dim, self.ft * 16)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(self.ft * 32, self.ft * 16),
                Conv3DBlock(self.ft * 16, self.ft * 16),
                Conv3DBlock(self.ft * 16, self.ft * 16),
                SingleDeconv3DBlock(self.ft * 16, self.ft * 8)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(self.ft * 16, self.ft * 8),
                Conv3DBlock(self.ft * 8, self.ft * 8),
                SingleDeconv3DBlock(self.ft * 8, self.ft * 4)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(self.ft * 8, self.ft * 4),
                Conv3DBlock(self.ft * 4, self.ft * 4),
                SingleDeconv3DBlock(self.ft * 4, self.ft * 2)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(self.ft * 4, self.ft * 2),
                Conv3DBlock(self.ft * 2, self.ft * 2),
                SingleConv3DBlock(self.ft * 2, output_dim, 1)
            )

    def forward(self, image, hidden_state):
        z0 = image
        z3, z6, z9, z12  = torch.unbind(hidden_state, dim=1)

        z0 = self.downsample_z0(z0)
        for i, downsample in enumerate(self.downsample):
            if i == 0:
                z3 = downsample(z3)
            elif i == 1:
                z6 = downsample(z6)
            elif i == 2:
                z9 = downsample(z9)
            else:
                z12 = downsample(z12)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        z0 = F.interpolate(z0, size=(192, 192, 192), mode='trilinear', align_corners=False)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return F.softmax(output, dim=1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNETR()
    model.to(device)

    output = model(torch.randn(1, 1, 480, 480, 240).to(device), torch.randn(1, 4, 512, 24, 24, 24).to(device))
