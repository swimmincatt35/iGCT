#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=180g # 180g
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --job-name=launch_igct_c10
#SBATCH --output=launch_igct_c10_%j.out
#SBATCH --error=launch_igct_c10_%j.err

module load cuda/12.1.1
source igct/bin/activate

NODES=1
NGPUS=8
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 55535))

srun --nodes=1 --ntasks-per-node=1 torchrun --nnodes=$NODES --nproc_per_node=$NGPUS --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    igct_train.py  \
    --outdir=igct-runs \
    --desc reproduce_c10 \
    --duration=368.64 --tick=102.4 --batch=1024 --lr=0.0001 --optim=RAdam --dropout=0.2 --precond=igct --cond=1 --guidance=1 --arch=ncsnpp \
    --recon=2e-5 --recon_every=10 --data=datasets/cifar10-32x32.zip --test_data=datasets/cifar10-32x32-test.zip \
    --eval_every=400 --sample_every=100 --dump=400 --snap=800 \
    -c 0.03 --double=400 --w_embed_dim=256 --w_min=0 --w_max=14 \
    --guide_t_low=11.0 --guide_t_high=14.3 --guide_p_pw=2.0 --guide_p_max=0.9 \
    --metrics=pr50k3_full,fid50k_full,recon_full,cifar10_edit_full \
    --enable_tf32=1
