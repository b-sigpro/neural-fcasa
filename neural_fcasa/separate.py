from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
import pickle as pkl

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc

from einops import rearrange
import numpy as np

import torch
from torch import nn
from torch.nn import functional as fn  # noqa

from einops.layers.torch import Rearrange
from torchaudio.transforms import InverseSpectrogram

from huggingface_hub import snapshot_download
from kornia.filters import MedianBlur

import soundfile as sf

from neural_fcasa.utils.separator import main


@dataclass
class Context:
    model: nn.Module
    istft: nn.Module
    median_filt: nn.Module
    config: ListConfig | DictConfig


def add_common_args(parser):
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--out_ch", type=int, default=0)
    parser.add_argument("--medfilt_size", type=int, default=11)
    parser.add_argument("--noi_snr", type=float, default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--dump_diar", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")


def initialize(args: Namespace, unk_args: list[str]):
    if args.model_path.startswith("hf://"):
        hf_path = str(args.model_path).removeprefix("hf://")
        args.model_path = Path(snapshot_download(hf_path))
    else:
        args.model_path = Path(args.model_path)

    config = oc.merge(
        oc.load(Path(args.model_path) / "config.yaml"),  # todo
        oc.from_cli(unk_args),
    )

    config.autocast = args.device.startswith("cuda")

    checkpoint_path = Path(args.model_path) / "version_0" / "checkpoints" / "last.ckpt"
    config.task._target_ += ".load_from_checkpoint"
    model = instantiate(
        config.task,
        checkpoint_path=checkpoint_path,
        map_location=args.device,
    )
    model.eval()

    istft = InverseSpectrogram(model.stft[0].n_fft, hop_length=model.stft[0].hop_length).to(args.device)

    median_filt = nn.Sequential(
        Rearrange("b n t -> b n 1 t"),
        MedianBlur((1, args.medfilt_size)),
        Rearrange("b n 1 t -> b n t"),
    ).to(args.device)
    median_filt.eval()

    return Context(model, istft, median_filt, config)


def separate(src_filename: Path, dst_filename: Path, ctx: Context, args: Namespace, unk_args: list[str]):
    model, istft = ctx.model, ctx.istft

    # load wav
    src_wav, sr = sf.read(src_filename, dtype=np.float32)
    src_wav = rearrange(torch.from_numpy(src_wav).to(model.device), "t m -> 1 m t")

    # calculate spectrogram
    xraw = model.stft(src_wav)[..., : src_wav.shape[-1] // model.hop_length]  # [B, F, M, T]
    scale = xraw.abs().square().clip(1e-6).mean(dim=(1, 2, 3), keepdims=True).sqrt()
    x = xraw / scale

    # encode
    z, w, g, Q, xt = model.encoder(x)
    w = ctx.median_filt(w).gt(args.thresh).to(torch.float32)

    # decode
    lm = model.decoder(z)  # [B, F, N, T]

    # Wiener filtering
    yt = torch.einsum("bnt,bfnt,bfmn->bfmt", w, lm, g).add(1e-6)
    Qx_yt = torch.einsum("bfmn,bfnt->bfmt", Q, xraw) / yt
    s = torch.einsum("bfm,bnt,bfnt,bfmn,bfmt->bnft", torch.linalg.inv(Q)[:, :, args.out_ch], w, lm, g, Qx_yt)

    dst_wav = istft(s, src_wav.shape[-1])

    if args.dump_diar:
        with open(dst_filename.with_suffix(".diar"), "wb") as f:
            pkl.dump(w.cpu().numpy(), f)

    dst_wav = rearrange(dst_wav, "1 m t -> t m")

    if args.noi_snr is not None:
        scale = dst_wav.square().mean().sqrt().clip(1e-6) * 10 ** (-args.noi_snr / 20)
        dst_wav = dst_wav + torch.randn_like(dst_wav) * scale

    if args.normalize:
        dst_wav /= dst_wav.abs().max().clip(1e-6)

    # save separated signal
    sf.write(dst_filename, dst_wav.cpu().numpy(), sr, "PCM_24")


if __name__ == "__main__":
    main(add_common_args, initialize, separate)
