from argparse import ArgumentParser
from pathlib import Path

from nara_wpe.wpe import wpe

import librosa as lr
import soundfile as sf


def main():
    parser = ArgumentParser()
    parser.add_argument("src_filename", type=Path)
    parser.add_argument("dst_filename", type=Path)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--taps", type=int, default=10)
    parser.add_argument("--delay", type=int, default=3)
    args = parser.parse_args()

    wav_src, sr = sf.read(args.src_filename)
    assert sr == 16000

    x = lr.stft(wav_src.T, n_fft=args.n_fft, hop_length=args.hop_length)
    y = wpe(x.transpose(1, 0, 2), taps=args.taps, delay=args.delay).transpose(1, 0, 2)

    wav_dst = lr.istft(y, hop_length=args.hop_length, length=wav_src.shape[0])
    sf.write(args.dst_filename, wav_dst.T, sr, "PCM_24")


if __name__ == "__main__":
    main()
