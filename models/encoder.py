import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: Float[Tensor, "batch in_ch h w"]) -> Float[Tensor, "batch out_ch h_out w_out"]:
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(F.relu(x))
        return x


class ProtoNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        flatten: bool = True,
    ) -> None:
        super().__init__()

        layers: list[ConvBlock] = []
        channels_in = in_channels
        for _ in range(num_blocks):
            layers.append(ConvBlock(channels_in, hidden_channels))
            channels_in = hidden_channels

        self.net = nn.Sequential(*layers)
        self.flatten = flatten

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Tensor:
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
    encoder.eval()
    with torch.no_grad():
        x = torch.zeros(1, in_channels, image_size, image_size, device=device)
        z = encoder(x)
        return int(z.shape[1])
