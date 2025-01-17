#!/usr/bin/env python3

from argparse import ArgumentParser
from math import ceil, floor
import os
from pathlib import Path

from progressbar import ProgressBar

import numpy as np
import pandas as pd

import soundfile as sf


class EmptySample(Exception):
    pass


def make_dataset(args, unk_args):
    import h5py
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mode_path = Path(f"./{args.mode}/")

    if rank == 0:
        print("================================")
        print("Parameters")
        print("--------------------------------")
        for key, val in args.__dict__.items():
            print(f"{key:20s}: {val}")
        print("================================")

    # obtain file list
    print("init")
    if rank == 0:
        wav_fname_list = sorted((mode_path / args.data).glob("*.wav"))

        n_fname = len(wav_fname_list)
        wav_fname_list += (ceil(n_fname / size) * size - n_fname) * [None]
    else:
        wav_fname_list = None
    wav_fname_list = comm.bcast(wav_fname_list, root=0)

    label_resolution = 16000 / args.hop_length

    hdf_name = f"hdf5/chunk.{args.data}-hop{args.hop_length}-{args.mode}.hdf5"
    with h5py.File(hdf_name, "w", driver="mpio", comm=comm) as f:
        pbar = ProgressBar(redirect_stdout=True) if rank == 0 else lambda x: x
        for widx, wav_fname in enumerate(pbar(wav_fname_list[rank::size])):
            try:
                if wav_fname is None:
                    raise EmptySample()

                # load spectrogram
                wav, sr = sf.read(wav_fname, dtype=np.float32)
                duration, n_mic = wav.shape

                csv_fname = mode_path / "act" / f"{wav_fname.stem}.csv"
                if not csv_fname.exists():
                    raise EmptySample()

                df = pd.read_csv(csv_fname, names=("transcriber_start", "transcriber_end", "speaker_idx"))

                act = np.zeros([5, duration // args.hop_length], dtype=np.float32)
                for _, (start, end, spk) in df.iterrows():
                    act[int(spk), floor(start * label_resolution) : ceil(end * label_resolution)] = 1

                grp_name = f"{size*widx + rank:08d}"
            except EmptySample:
                grp_name, duration = None, None
            except Exception as e:
                print(f"Error!!!: {wav_fname} has error: {e}.")
                grp_name, duration = None, None

            all_grp_names = filter(None, comm.allgather(grp_name))
            all_durations = filter(None, comm.allgather(duration))
            for grp_name_, duration_ in zip(all_grp_names, all_durations, strict=False):
                g = f.create_group(grp_name_)
                g.create_dataset("wav", [n_mic, duration_], "float32")
                g.create_dataset("act", [5, duration_ // args.hop_length], "float32")

            # store data
            if grp_name is not None:
                f[f"{grp_name}/wav"][:] = wav.T  # type: ignore
                f[f"{grp_name}/act"][:] = act  # type: ignore


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

    for mode in ["tr", "cv"]:
        fname_job = f"jobs/{command_name}/{mode}.sh"
        fname_stdout = f"jobs.out/{command_name}/{mode}.out"

        with open(fname_job, "w") as f:
            f.write(job_template)

            f.write("mpirun -np 160 -npernode 40 --hostfile $SGE_JOB_HOSTLIST ")
            f.write("-mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 ")
            f.write(f"singularity exec --nv {dataset_path}/../singularity/singularity.sif direnv exec . ")
            f.write(f" python ./scripts/{command_name}.py gen --mode {mode} ")
            f.write(" ".join(unk_args) + "\n")

        os.system(f"qsub -g $JOB_GROUP $QSUB_ARGS -l rt_F=4 -l h_rt=1:0:0 -o {fname_stdout} {fname_job}")


def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser("gen", help="generate hdf5")
    sub_parser.add_argument("--mode", type=str, default="tr")
    sub_parser.add_argument("--data", type=str, default="derev")
    sub_parser.add_argument("--hop_length", type=int, default=160)
    sub_parser.set_defaults(handler=make_dataset)

    sub_parser = sub_parsers.add_parser("sub", help="submit jobs")
    sub_parser.set_defaults(handler=submit_jobs)

    args, unk_args = parser.parse_known_args()
    if hasattr(args, "handler"):
        args.handler(args, unk_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
