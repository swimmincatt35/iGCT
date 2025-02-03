#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=180g # 180g
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --partition=3090-gcondo
#SBATCH --job-name=test_eval_igct_im64_few_step
#SBATCH --output=test_eval_igct_im64_few_step_%j.out
#SBATCH --error=test_eval_igct_im64_few_step_%j.err
#SBATCH --constraint=a5000
#SBATCH --exclude=gpu2601

module load cuda/12.1.1
source igct/bin/activate

NODES=1
NGPUS=8
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 55535))


srun --nodes=1 --ntasks-per-node=1 \
    torchrun --nnodes=$NODES --nproc_per_node=$NGPUS --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    igct_eval.py \
    --outdir=igct-eval-runs \
    --desc eval_igct_im64 \
    --batch=64 --precond=igct --arch=adm-small \
    --data=datasets/imagenet-64x64.zip \
    --test_data=datasets/imagenet-64x64-val.zip \
    --im64_subg_dir=datasets/imagenet-64x64-editing-subgroups \
    --cond=1 --w_embed_dim=256 --few_step=1 \
    --metrics=pr50k3_full,fid50k_full,recon_full,im64_edit_full \
    --net_pkl=/path/to/igct/pkl
