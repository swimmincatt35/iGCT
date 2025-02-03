#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=180g
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH --job-name=eval_guided_cd_c10
#SBATCH --output=eval_guided_cd_c10_%j.out
#SBATCH --error=eval_guided_cd_c10_%j.err

module load cuda/12.1.1
source igct/bin/activate

NODES=1
NGPUS=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 55535))


srun --nodes=1 --ntasks-per-node=1 \
    torchrun --nnodes=$NODES --nproc_per_node=$NGPUS --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    igct_eval.py \
    --outdir=igct-eval-runs \
    --desc eval_guided_cd_c10 \
    --batch=64 --precond=cd --arch=ncsnpp \
    --data=datasets/cifar10-32x32.zip \
    --test_data=datasets/cifar10-32x32-test.zip \
    --cond=1 --w_embed_dim=256 --few_step=1 \
    --metrics=pr50k3_full,fid50k_full \
    --net_pkl=/path/to/guided-cd/pkl

