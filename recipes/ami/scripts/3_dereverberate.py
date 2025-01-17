#! /usr/bin/env python3

from argparse import ArgumentParser
from functools import partial
import os
from pathlib import Path

from progressbar import progressbar as pbar

import cupy as cp
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from wpe import wpe

import librosa as lr
import soundfile as sf


def split_data_one(src_filename, dst_path):
    """
    Note that we used our torch implementation of WPE in our Interspeech paper,
    while here we replaced it with the more standard `gpu-wpe`.
    If there is a reproduction issue, please let us know.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    with cp.cuda.Device(rank % 4):
        src_wav, sr = sf.read(src_filename)

        src_spec = lr.stft(src_wav.T, n_fft=512, hop_length=160)  # [M, F, T]
        src_spec = cp.asarray(src_spec)
        M, F, T = src_spec.shape

        if (cp.abs(src_spec) ** 2).max(axis=0).min() == 0:
            return

        dst_spec = wpe(src_spec, taps=10, delay=3)

        dst_wav = lr.istft(dst_spec.get().transpose(1, 0, 2), hop_length=160).T

    sf.write(dst_path / src_filename.name, dst_wav, sr, "PCM_24")


def split_data(args, unk_args):
    src_filename_list = list((Path(f"./{args.mode}") / "mix").glob("*.wav"))

    dst_path = Path(f"./{args.mode}") / "derev"
    dst_path.mkdir(parents=True, exist_ok=True)

    with MPIPoolExecutor() as pool:
        func = partial(split_data_one, dst_path=dst_path)
        for _ in pbar(pool.map(func, src_filename_list), max_value=len(src_filename_list)):
            pass


def submit_jobs(args, unk_args):
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent
    command_name = script_path.stem

    job_path = Path(f"jobs/{command_name}/")
    out_path = Path(f"jobs.out/{command_name}/")
    job_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(f"{dataset_path}/scripts/job_template.sh") as f:
        job_template = f.read()

    for mode in ["tr", "cv", "tt"]:
        filename_job = job_path / f"{mode}.sh"
        filename_stdout = out_path / f"{mode}.out"

        with open(filename_job, "w") as f:
            f.write(job_template)

            f.write("mpirun -np 64 -npernode 4 --hostfile $SGE_JOB_HOSTLIST ")
            f.write("-mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 ")
            f.write(f"singularity exec --nv {dataset_path}/../singularity/singularity.sif direnv exec . ")
            f.write(f" python -m mpi4py.futures ./scripts/{command_name}.py job --mode {mode} ")
            f.write(" ".join(unk_args) + "\n")

        os.system(f"qsub -g $JOB_GROUP $QSUB_ARGS -l rt_F=16 -l h_rt=3:0:0 -o {filename_stdout} {filename_job}")


def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser("job", help="dereverberate mixture signals")
    sub_parser.add_argument("--mode", type=str, default="tr")
    sub_parser.set_defaults(handler=split_data)

    sub_parser = sub_parsers.add_parser("sub", help="submit jobs")
    sub_parser.set_defaults(handler=submit_jobs)

    args, unk_args = parser.parse_known_args()
    if hasattr(args, "handler"):
        args.handler(args, unk_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
