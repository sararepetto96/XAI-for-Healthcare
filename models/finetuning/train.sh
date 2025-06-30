#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

set -x
GPUS=$1

exp_name=medvit_cls_exp
time=$(date "+%m%d_%H%M%S")
save_root_dir=../${exp_name}/${time}

if [ ! -d ${save_root_dir} ]; then
    mkdir -p ${save_root_dir}
    echo save root dir is ${save_root_dir}.
else
    echo Error, save root dir ${save_root_dir} exists, please run the shell again!
    exit 1
fi

# Pick a random port
if [ -z "$SLURM_JOBID" ]; then
    echo "SLURM_JOBID is not set, using a random port."
    export MASTER_PORT=$(expr 10000 + $RANDOM % 1000)
else
    echo "SLURM_JOBID is set to $SLURM_JOBID, using it to determine the port."
    export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
fi

export MASTER_ADDR=localhost  # Ensures it's bound to local machine

echo $MASTER_ADDR $MASTER_PORT

python3 -m torch.distributed.launch --nproc_per_node=$GPUS --use_env --master-port=$MASTER_PORT $parent_path/main.py \
--output-dir ${save_root_dir} \
--dist-eval ${@:2} \
