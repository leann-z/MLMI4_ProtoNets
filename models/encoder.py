# Encoder: 4-block CNN backbone for ProtoNet (Snell et al. 2017)
# Each block: Conv2d -> BatchNorm -> ReLU -> MaxPool

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class ConvBlock(nn.Module):
    # single conv block: Conv -> BN -> ReLU -> MaxPool

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        return x
    
class ProtoNetEncoder(nn.Module):
    # stacks num_blocks ConvBlocks, all with hidden_channels filters
    # output is [B, D] flat embedding by default (flatten=True)

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        flatten: bool = True,
    ):
        super().__init__()

        if num_blocks != 4:
            warnings.warn(f"Paper uses 4 blocks; got {num_blocks}")

        layers = []
        c_in = in_channels
        for _ in range(num_blocks):
            layers.append(ConvBlock(c_in, hidden_channels))
            c_in = hidden_channels

        self.net = nn.Sequential(*layers)
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, D]

        z = self.net(x)
        if self.flatten:
            z = torch.flatten(z, start_dim=1)
        return z


def infer_embedding_dim(
    encoder: nn.Module,
    in_channels: int,
    image_size: int,
    device: str | torch.device = "cpu",
    ) -> int:
    # pass a dummy input through to get embedding dim without hardcoding
    encoder.eval()
    with torch.no_grad():
        x = torch.zeros(1, in_channels, image_size, image_size, device=device)
        z = encoder(x)
        return int(z.shape[1])