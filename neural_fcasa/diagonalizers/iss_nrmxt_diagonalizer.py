import torch
from torch import nn


class ISSDiagonalizer(nn.Module):
    def __init__(self, n_iter: int = 1, eps: float = 1e-6, eps2: float = 1e-6):
        super().__init__()

        self.n_iter = n_iter

        self.eps = eps
        self.eps2 = eps2

    def forward(self, r, Q, x):
        """
        Parameters
        ----------
        r : (B, F, M, T) Tensor
        Q : (B, F, M, M) Tensor
        x : (B, F, M, T) Tensor
        """

        _, _, M, T = x.shape

        Qx = torch.einsum("...mn,...nt->...mt", Q, x)
        xt = Qx.real**2 + Qx.imag**2  # torch.abs(Qx) ** 2

        V = torch.einsum("...kt,...mt,...nt->...kmn", r, x, x.conj()) / T
        V = V + self.eps * torch.eye(M, device=x.device)

        for _ in range(self.n_iter):
            for k in range(M):
                q = Q[..., k, :]
                Vq = torch.einsum("...kmn,...n->...km", V, q.conj())

                qVq = torch.einsum("...m,...km->...k", q, Vq).real.clip(self.eps2)
                v = torch.einsum("...km,...km->...k", Q, Vq) / qVq.to(x.dtype)
                v[..., k] = 1 - qVq[..., k] ** -0.5

                Q = Q - torch.einsum("...m,...n->...mn", v, q)

            Qx = torch.einsum("...mn,...nt->...mt", Q, x)
            xt = Qx.real**2 + Qx.imag**2  # torch.abs(Qx) ** 2

            scale = xt.mean(dim=(1, 2, 3), keepdim=True)
            xt = xt / scale
            Q = Q / scale.clip(1e-6).sqrt().to(x.dtype)

        return Q, xt
