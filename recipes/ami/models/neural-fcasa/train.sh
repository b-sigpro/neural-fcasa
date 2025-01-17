#! /bin/bash

#$-l rt_F=16
#$-l h_rt=24:0:00
#$-j y
#$-cwd

if [ "$1" = "-q" ]; then
    cwd=`dirname "${0}"`

    rm -i $cwd/log.txt
    rm -r $cwd/version_*

    qsub $QSUB_ARGS -g $JOB_GROUP -o "$cwd/log.txt" "${0}" "$cwd"
else
    if [ -z "$1" ]; then
        wd=`dirname "${0}"`
    else
        wd="${1}"
    fi

    echo JOB_ID: $JOB_ID

    singularity_path=./singularity/singularity.sif

    source /etc/profile.d/modules.sh
    module load singularitypro
    module load hpcx/2.12

    ngpu=$(nvidia-smi -L | wc -l)

    mpirun -np $((NHOSTS*ngpu)) -npernode $ngpu --hostfile $SGE_JOB_HOSTLIST \
        -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 \
        -x MAIN_ADDR=$(hostname -i)  \
        -x MAIN_PORT=3000 \
        -x SINGULARITY_BINDPATH="/groups,/scratch" \
        -x COLUMNS=120 \
        -x PYTHONUNBUFFERED=true \
        singularity exec --nv $singularity_path direnv exec . \
        python -m aiaccel.apps.train $wd/config.yaml --working_directory $wd
fi
