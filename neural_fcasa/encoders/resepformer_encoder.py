from einops import rearrange, repeat

import torch  # noqa
from torch import nn
from torch.distributions import Normal
from torch.distributions.utils import logits_to_probs
from torch.nn import functional as fn

from einops.layers.torch import Rearrange, Reduce

from neural_fcasa.nn import (
    MakeChunk,
    OverlappedAdd,
    PositionalEncoding,
    SpatialPositionalEncoding,
    TACModule,
)
from neural_fcasa.utils.distributions import ApproxBernoulli


class RESepFormerBlock(nn.Module):
    def __init__(
        self,
        n_mic: int,
        n_layers: int,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        def tf_generator():
            return nn.TransformerEncoderLayer(
                d_model,
                n_head,
                dim_feedforward,
                batch_first=True,
                norm_first=norm_first,
                layer_norm_eps=layer_norm_eps,
            )

        self.intra_tf_list = nn.ModuleList([tf_generator() for _ in range(n_layers)])
        self.inter_tf_list = nn.ModuleList([tf_generator() for _ in range(n_layers)])

        self.tac_list = nn.ModuleList([TACModule(n_mic, d_model) for _ in range(n_layers)])

        self.last_intra_tf = tf_generator()
        self.gln = nn.GroupNorm(1, d_model)

    def forward(self, x: torch.Tensor):
        B, S, T, C = x.shape

        h = rearrange(x, "b s t c -> (b s) t c")
        for intra_tf, inter_tf, tac in zip(self.intra_tf_list, self.inter_tf_list, self.tac_list, strict=False):
            # intra-chunk
            h = intra_tf(h)

            # inter-mic (tac)
            h = rearrange(tac(rearrange(h, "(b s) t c -> b (s t) c", s=S)), "b (s t) c -> (b s) t c", s=S)

            # inter-chunk
            h = h + repeat(inter_tf(rearrange(h.mean(dim=1), "(b s) c -> b s c", s=S)), "b s c -> (b s) t c", t=T)

        # intra-chunk
        h = rearrange(self.last_intra_tf(h), "(b s) t c -> b s t c", s=S)
        h = rearrange(self.gln(rearrange(h, "b s t c -> b c (s t)")), "b c (s t) -> b s t c", s=S)

        return h


class ReSepFormerModule(nn.Module):
    def __init__(
        self,
        n_mic: int,
        n_stft: int,
        chunk_size: int,
        step_size: int,
        n_layers: int,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
        use_r: bool = True,
        autocast: bool = True,
        spec_aug: nn.Module | None = None,
    ):
        super().__init__()

        self.spec_aug = nn.Sequential(
            Rearrange("b f m t -> b m f t"),
            spec_aug if spec_aug is not None else nn.Identity(),
            Rearrange("b m f t -> b f m t"),
        )

        self.bn0 = nn.BatchNorm2d(n_stft)
        self.lin0 = nn.Sequential(
            Rearrange("b c m t -> (b m) t c"),
            nn.Linear(n_stft, d_model),
            PositionalEncoding(d_model),
            #
            Rearrange("(b m) t c -> b m t c", m=n_mic),
            SpatialPositionalEncoding(n_mic, d_model),
            Rearrange("b m t c -> (b m) t c"),
            #
            MakeChunk(chunk_size, step_size),  # [B, S, T, C]
        )

        self.lin1 = nn.Linear(2 * d_model, d_model)

        self.enc = RESepFormerBlock(n_mic, n_layers, d_model, n_head, dim_feedforward, norm_first, layer_norm_eps)

        if use_r:
            self.overlapped_add = OverlappedAdd(chunk_size, step_size)
            self.head_r = nn.Sequential(
                nn.Linear(d_model, n_stft),
                nn.Sigmoid(),
                Rearrange("(b m) t f -> b f m t", m=n_mic),
            )
        else:
            self.overlapped_add = lambda *_: None
            self.head_r = lambda *_: None

        self.autocast = autocast

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        _, _, M, T = x.shape

        logx = self.bn0(x.clip(1e-6).log())
        logx = self.spec_aug(logx) if self.training else logx

        h = self.lin0(logx)

        if h0 is None:
            h0 = torch.zeros_like(h)

        with torch.autocast("cuda", dtype=torch.float16, enabled=self.autocast):
            h = self.enc(self.lin1(torch.concat((h, h0), dim=-1)))
        h = h.to(torch.float32)

        return h + h0, self.head_r(self.overlapped_add(h, T))


class RESepFormerEncoder(nn.Module):
    def __init__(
        self,
        n_fft: int,
        n_mic: int,
        n_src: int,
        dim_latent: int,
        chunk_size: int,
        step_size: int,
        diagonalizer: nn.Module,
        n_blocks: int,
        n_layers: int = 1,
        d_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 2048,
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
        tau: float = 1.0,
        autocast: bool = True,
        use_jit: bool = True,
        spec_aug: nn.Module | None = None,
    ):
        super().__init__()

        n_stft = n_fft // 2 + 1

        self.n_blocks = n_blocks

        self.tau = tau

        self.tf = nn.ModuleList()
        for _ in range(n_blocks):
            self.tf.append(
                ReSepFormerModule(
                    n_mic,
                    n_stft,
                    chunk_size,
                    step_size,
                    n_layers,
                    d_model,
                    n_head,
                    dim_feedforward,
                    norm_first,
                    layer_norm_eps,
                    _ < n_blocks - 1,
                    autocast,
                    spec_aug,
                )
            )

        self.diagonalizer = torch.jit.script(diagonalizer) if use_jit else diagonalizer

        self.overlapped_add = OverlappedAdd(chunk_size, step_size)

        self.head_z_val = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.Linear(d_model, 2 * dim_latent),
            Rearrange("(b m) t (c d) -> c b d m t", m=n_mic, c=2, d=dim_latent),
        )

        self.head_w_val = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.Linear(d_model, 1),
            Rearrange("(b m) t 1 -> b m t", m=n_mic),
        )

        self.head_zw_att = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            nn.Linear(d_model, n_src),
            Reduce("(b m) t n -> b m n", "mean", m=n_mic),
            nn.Softmax(dim=-2),
        )

        self.head_g = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.PReLU(),
            #
            Reduce("b t d -> b d", "mean"),
            #
            nn.Linear(d_model, n_stft * n_src),
            Rearrange("(b m) (f n) -> b f m n", m=n_mic, f=n_stft),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor, distribution: bool = False):
        B, F, M, T = x.shape

        h, r = self.tf[0](x.abs().square())
        xt, Q = None, repeat(torch.eye(M, dtype=torch.complex64, device=x.device), "m n -> b f m n", b=B, f=F)
        for tf_ in self.tf[1:]:  # type: ignore
            Q, xt = self.diagonalizer(r, Q, x)

            h, r = tf_(xt, h)  # [BxM, T, C]

        h = self.overlapped_add(h, T)

        zw_att = self.head_zw_att(h)  # [B, D, N, T]

        # z
        z_mu_, z_sig_ = self.head_z_val(h)  # [B, D, N, T]

        z_mu: torch.Tensor = torch.einsum("bmn,bdmt->bdnt", zw_att, z_mu_)
        w_logits = torch.einsum("bmn,bmt->bnt", zw_att, self.head_w_val(h))

        if distribution:
            qz = Normal(z_mu, fn.softplus(torch.einsum("bmn,bdmt->bdnt", zw_att, z_sig_)) + 1e-6)
            qw = ApproxBernoulli(logits=w_logits, temperature=torch.full_like(w_logits, self.tau))
        else:
            qz = z_mu
            qw = logits_to_probs(w_logits, is_binary=True)

        # g
        g: torch.Tensor = self.head_g(h) + 1e-6  # type: ignore

        return qz, qw, g, Q, xt
