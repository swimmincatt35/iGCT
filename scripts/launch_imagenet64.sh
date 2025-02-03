#!/bin/bash
#SBATCH --nodes=8
#SBATCH --mem=180g 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=240:00:00
#SBATCH --partition=gpu-he
#SBATCH --job-name=launch_igct_im64
#SBATCH --output=launch_igct_im64_%j.out
#SBATCH --error=launch_igct_im64_%j.err

module load cuda/12.1.1
source igct/bin/activate

# Recommended resources: 64 A100 (80G) gpus, training for 260k ticks

NODES=8 
NGPUS=8
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 55535))

srun --nodes=1 --ntasks-per-node=1 torchrun --nnodes=$NODES --nproc_per_node=$NGPUS --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    igct_train.py  \
    --outdir=igct-runs \
    --desc reproduce_im64 \
    --duration=266.24 --tick=102.4 --batch=1024 --batch-gpu=16 --lr=0.0001 --optim=RAdam --dropout=0.3 --precond=igct --cond=1 --guidance=1 --arch=adm-small \
    --recon=2e-5 --recon_sch=1 --recon_every=10 --data=datasets/imagenet-64x64.zip --test_data=datasets/imagenet-64x64-val.zip \
    --im64_subg_dir=datasets/imagenet-64x64-editing-subgroups \
    --eval_every=400 --sample_every=100 --dump=400 --snap=800 \
    -c 0.06 --double=400 --w_embed_dim=256 --w_min=0 --w_max=14 \
    --guide_t_low=11.0 --guide_t_high=14.3 --guide_p_pw=2.0 --guide_p_max=0.9 \
    --metrics=pr50k3_full,fid50k_full,recon_full,im64_edit_full \
    --enable_tf32=1 &

srun --nodes=1 --ntasks-per-node=1 torchrun --nnodes=$NODES --nproc_per_node=$NGPUS --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    igct_train.py  \
    --outdir=igct-runs \
    --desc reproduce_im64 \
    --duration=266.24 --tick=102.4 --batch=1024 --batch-gpu=16 --lr=0.0001 --optim=RAdam --dropout=0.3 --precond=igct --cond=1 --guidance=1 --arch=adm-small \
    --recon=2e-5 --recon_sch=1 --recon_every=10 --data=datasets/imagenet-64x64.zip --test_data=datasets/imagenet-64x64-val.zip \
    --im64_subg_dir=datasets/imagenet-64x64-editing-subgroups \
    --eval_every=400 --sample_every=100 --dump=400 --snap=800 \
    -c 0.06 --double=400 --w_embed_dim=256 --w_min=0 --w_max=14 \
    --guide_t_low=11.0 --guide_t_high=14.3 --guide_p_pw=2.0 --guide_p_max=0.9 \
    --metrics=pr50k3_full,fid50k_full,recon_full,im64_edit_full \
    --enable_tf32=1 &

srun --nodes=1 --ntasks-per-node=1 torchrun --nnodes=$NODES --nproc_per_node=$NGPUS --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    igct_train.py  \
    --outdir=igct-runs \
    --desc reproduce_im64 \
    --duration=266.24 --tick=102.4 --batch=1024 --batch-gpu=16 --lr=0.0001 --optim=RAdam --dropout=0.3 --precond=igct --cond=1 --guidance=1 --arch=adm-small \
    --recon=2e-5 --recon_sch=1 --recon_every=10 --data=datasets/imagenet-64x64.zip --test_data=datasets/imagenet-64x64-val.zip \
    --im64_subg_dir=datasets/imagenet-64x64-editing-subgroups \
    --eval_every=400 --sample_every=100 --dump=400 --snap=800 \
    -c 0.06 --double=400 --w_embed_dim=256 --w_min=0 --w_max=14 \
    --guide_t_low=11.0 --guide_t_high=14.3 --guide_p_pw=2.0 --guide_p_max=0.9 \
    --metrics=pr50k3_full,fid50k_full,recon_full,im64_edit_full \
    --enable_tf32=1 &

srun --nodes=1 --ntasks-per-node=1 torchrun --nnodes=$NODES --nproc_per_node=$NGPUS --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    igct_train.py  \
    --outdir=igct-runs \
    --desc reproduce_im64 \
    --duration=266.24 --tick=102.4 --batch=1024 --batch-gpu=16 --lr=0.0001 --optim=RAdam --dropout=0.3 --precond=igct --cond=1 --guidance=1 --arch=adm-small \
    --recon=2e-5 --recon_sch=1 --recon_every=10 --data=datasets/imagenet-64x64.zip --test_data=datasets/imagenet-64x64-val.zip \
    --im64_subg_dir=datasets/imagenet-64x64-editing-subgroups \
    --eval_every=400 --sample_every=100 --dump=400 --snap=800 \
    -c 0.06 --double=400 --w_embed_dim=256 --w_min=0 --w_max=14 \
    --guide_t_low=11.0 --guide_t_high=14.3 --guide_p_pw=2.0 --guide_p_max=0.9 \
    --metrics=pr50k3_full,fid50k_full,recon_full,im64_edit_full \
    --enable_tf32=1 &

srun --nodes=1 --ntasks-per-node=1 torchrun --nnodes=$NODES --nproc_per_node=$NGPUS --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    igct_train.py  \
    --outdir=igct-runs \
    --desc reproduce_im64 \
    --duration=266.24 --tick=102.4 --batch=1024 --batch-gpu=16 --lr=0.0001 --optim=RAdam --dropout=0.3 --precond=igct --cond=1 --guidance=1 --arch=adm-small \
    --recon=2e-5 --recon_sch=1 --recon_every=10 --data=datasets/imagenet-64x64.zip --test_data=datasets/imagenet-64x64-val.zip \
    --im64_subg_dir=datasets/imagenet-64x64-editing-subgroups \
    --eval_every=400 --sample_every=100 --dump=400 --snap=800 \
    -c 0.06 --double=400 --w_embed_dim=256 --w_min=0 --w_max=14 \
    --guide_t_low=11.0 --guide_t_high=14.3 --guide_p_pw=2.0 --guide_p_max=0.9 \
    --metrics=pr50k3_full,fid50k_full,recon_full,im64_edit_full \
    --enable_tf32=1 &

srun --nodes=1 --ntasks-per-node=1 torchrun --nnodes=$NODES --nproc_per_node=$NGPUS --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    igct_train.py  \
    --outdir=igct-runs \
    --desc reproduce_im64 \
    --duration=266.24 --tick=102.4 --batch=1024 --batch-gpu=16 --lr=0.0001 --optim=RAdam --dropout=0.3 --precond=igct --cond=1 --guidance=1 --arch=adm-small \
    --recon=2e-5 --recon_sch=1 --recon_every=10 --data=datasets/imagenet-64x64.zip --test_data=datasets/imagenet-64x64-val.zip \
    --im64_subg_dir=datasets/imagenet-64x64-editing-subgroups \
    --eval_every=400 --sample_every=100 --dump=400 --snap=800 \
    -c 0.06 --double=400 --w_embed_dim=256 --w_min=0 --w_max=14 \
    --guide_t_low=11.0 --guide_t_high=14.3 --guide_p_pw=2.0 --guide_p_max=0.9 \
    --metrics=pr50k3_full,fid50k_full,recon_full,im64_edit_full \
    --enable_tf32=1 &

srun --nodes=1 --ntasks-per-node=1 torchrun --nnodes=$NODES --nproc_per_node=$NGPUS --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    igct_train.py  \
    --outdir=igct-runs \
    --desc reproduce_im64 \
    --duration=266.24 --tick=102.4 --batch=1024 --batch-gpu=16 --lr=0.0001 --optim=RAdam --dropout=0.3 --precond=igct --cond=1 --guidance=1 --arch=adm-small \
    --recon=2e-5 --recon_sch=1 --recon_every=10 --data=datasets/imagenet-64x64.zip --test_data=datasets/imagenet-64x64-val.zip \
    --im64_subg_dir=datasets/imagenet-64x64-editing-subgroups \
    --eval_every=400 --sample_every=100 --dump=400 --snap=800 \
    -c 0.06 --double=400 --w_embed_dim=256 --w_min=0 --w_max=14 \
    --guide_t_low=11.0 --guide_t_high=14.3 --guide_p_pw=2.0 --guide_p_max=0.9 \
    --metrics=pr50k3_full,fid50k_full,recon_full,im64_edit_full \
    --enable_tf32=1 &

srun --nodes=1 --ntasks-per-node=1 torchrun --nnodes=$NODES --nproc_per_node=$NGPUS --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    igct_train.py  \
    --outdir=igct-runs \
    --desc reproduce_im64 \
    --duration=266.24 --tick=102.4 --batch=1024 --batch-gpu=16 --lr=0.0001 --optim=RAdam --dropout=0.3 --precond=igct --cond=1 --guidance=1 --arch=adm-small \
    --recon=2e-5 --recon_sch=1 --recon_every=10 --data=datasets/imagenet-64x64.zip --test_data=datasets/imagenet-64x64-val.zip \
    --im64_subg_dir=datasets/imagenet-64x64-editing-subgroups \
    --eval_every=400 --sample_every=100 --dump=400 --snap=800 \
    -c 0.06 --double=400 --w_embed_dim=256 --w_min=0 --w_max=14 \
    --guide_t_low=11.0 --guide_t_high=14.3 --guide_p_pw=2.0 --guide_p_max=0.9 \
    --metrics=pr50k3_full,fid50k_full,recon_full,im64_edit_full \
    --enable_tf32=1 &

wait
