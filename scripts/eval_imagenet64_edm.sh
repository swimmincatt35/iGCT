#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=180g # 180g
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH --partition=gpu-he
#SBATCH --job-name=test_eval_edm_im64
#SBATCH --output=test_eval_edm_im64_%j.out
#SBATCH --error=test_eval_edm_im64_%j.err
#SBATCH --constraint=l40s
#SBATCH --exclude=gpu2601

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
    --desc eval_edm_im64 \
    --batch=64 --precond=edm --arch=adm \
    --data=datasets/imagenet-64x64.zip \
    --test_data=datasets/imagenet-64x64-val.zip \
    --im64_subg_dir=datasets/imagenet-64x64-editing-subgroups \
    --cond=1 \
    --metrics=pr50k3_full,fid50k_full,recon_full,im64_edit_full \
    --net_pkl=/path/to/edm/pkl
