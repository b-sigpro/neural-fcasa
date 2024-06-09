import torch  # noqa
from torch import nn

from einops.layers.torch import Rearrange


class ResConvBlock2d(nn.Sequential):
    def __init__(self, io_ch):
        super().__init__(
            nn.LayerNorm(io_ch),
            nn.Linear(io_ch, io_ch),
            nn.PReLU(),
            nn.Linear(io_ch, io_ch),
        )

    def forward(self, x):
        return x + super().forward(x)


class Decoder(nn.Module):
    def __init__(
        self,
        n_fft: int,
        dim_latent: int,
        io_ch: int = 256,
        n_layers: int = 3,
        dim_latent_noi: int | None = None,
    ):
        super().__init__()

        n_stft = n_fft // 2 + 1

        self.cnv = nn.Sequential(
            Rearrange("b d n t -> b n t d"),
            nn.Linear(dim_latent, io_ch),
            *[ResConvBlock2d(io_ch) for ll in range(n_layers - 1)],
            nn.Linear(io_ch, n_stft),
            nn.Softplus(),
            Rearrange("b n t f -> b f n t"),
        )

        self.dim_latent_noi = dim_latent_noi

    def forward(self, z):
        """
        Parameters
        ----------
        z : [B, D, N, T]
        """

        if self.dim_latent_noi is not None:
            z[:, self.dim_latent_noi :, -1, :] = 0.0

        return self.cnv(z) + 1e-6  # [B, F, N, T]
