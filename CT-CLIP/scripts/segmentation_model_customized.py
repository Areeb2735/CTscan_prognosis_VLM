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
            CustomSingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class CustomSingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(
            in_planes, 
            out_planes, 
            kernel_size=3, 
            stride=(2, 2, 1),  # Upsample first two dims, keep third the same
            padding=(1, 1, 1), 
            output_padding=(1, 1, 0)  # Adjusts output to exactly 12,12,6
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.deconv(x))

        
class UNETR(nn.Module):
    def __init__(self, input_dim=1, output_dim=3, embed_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(input_dim, 4, 3),
                Conv3DBlock(4, 8, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 64),
                Deconv3DBlock(64, 32),
                Deconv3DBlock(32, 16)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 64),
                Deconv3DBlock(64, 32),
            )

        self.decoder9 = \
            Deconv3DBlock(embed_dim, 64)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(embed_dim, 64)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                Conv3DBlock(64, 64),
                SingleDeconv3DBlock(64, 32)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(64, 32),
                Conv3DBlock(32, 32),
                SingleDeconv3DBlock(32, 16)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(32, 16),
                Conv3DBlock(16, 16),
                SingleDeconv3DBlock(16, 8)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(16, 8),
                Conv3DBlock(8, 8),
                SingleConv3DBlock(8, output_dim, 1)
            )

    def forward(self, image, hidden_state):
        z0 = image
        z3, z6, z9, z12  = torch.unbind(hidden_state, dim=1)

        # z0, z3, z6, z9, z12 = x, *z
        # z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        # z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        # z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        # z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        breakpoint()
        z0 = F.interpolate(z0, size=(384, 384, 384), mode='trilinear', align_corners=False)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNETR()
    model.to(device)

    output = model(torch.randn(1, 1, 480, 480, 240).to(device), torch.randn(1, 4, 512, 24, 24, 24).to(device))