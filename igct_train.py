import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import igct_training_loop as training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--test_data',     help='Path to the testing dataset', metavar='ZIP|DIR',             type=str, default=None) 
@click.option('--im64_subg_dir', help='Directory for im64 subgroups', metavar='DIR',                type=str, default=None)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture',                                       type=click.Choice(['ncsnpp','ddpmpp','ncsnpp-deep','ddpmpp-deep','adm-small','adm']), default='ncsnpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='igct',            type=click.Choice(['igct']), default='igct', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--optim',         help='Name of Optimizer', metavar='Optimizer',                     type=str, default='Adam', show_default=True)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=None, show_default=True)
@click.option('--ema_beta',      help='EMA decay rate', metavar='FLOAT',                            type=click.FloatRange(min=0), default=0.9999, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)
@click.option('--guidance',      help='Guided training', metavar='BOOL',                            type=bool, default=True, show_default=True)
@click.option('--w_embed_dim',   help='Dimension for w-embed ', metavar='INT',                      type=click.IntRange(min=1), default=256, show_default=True)

# Model Hyperparameters
@click.option('--mean',          help='P_mean of Log Normal Distribution', metavar='FLOAT',         type=click.FloatRange(), default=-1.1, show_default=True)
@click.option('--std',           help='P_std of Log Normal Distribution', metavar='FLOAT',          type=click.FloatRange(), default=2.0, show_default=True)
@click.option('--inv_mean',      help='inv_P_mean of Log Normal Distribution', metavar='FLOAT',     type=click.FloatRange(), default=-1.1, show_default=True)
@click.option('--inv_std',       help='inv_P_std of Log Normal Distribution', metavar='FLOAT',      type=click.FloatRange(), default=2.0, show_default=True)
@click.option('--double',        help='How often to reduce dt', metavar='TICKS',                    type=click.IntRange(min=1), default=500, show_default=True)

@click.option('-q',              help='Decay Factor', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=2.0, show_default=True)
@click.option('-c',              help='Constant c for Huber Loss', metavar='FLOAT',                 type=click.FloatRange(), default=0.0, show_default=True)
@click.option('-k',              help='Mapping fn hyperparams', metavar='FLOAT',                    type=click.FloatRange(), default=8.0, show_default=True)
@click.option('-b',              help='Mapping fn hyperparams', metavar='FLOAT',                    type=click.FloatRange(), default=1.0, show_default=True)
@click.option('--recon',         help='Reconstruction weight value.', metavar='FLOAT',              type=click.FloatRange(), default=1e-4, show_default=True)
@click.option('--recon_sch',     help='Reconstruction weight schedule.', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--recon_every',   help='How often to compute recon loss at training.', metavar='INT',type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--guide_t_low',   help='Guidance: guide_t_low. ', metavar='FLOAT',                   type=click.FloatRange(), default=11.0, show_default=True)
@click.option('--guide_t_high',  help='Guidance: guide_t_high. ', metavar='FLOAT',                  type=click.FloatRange(), default=14.3, show_default=True)
@click.option('--guide_p_pw',    help='Guidance: guide prob power.', metavar='FLOAT',               type=click.FloatRange(), default=2.0, show_default=True)
@click.option('--guide_p_max',   help='Guidance: guide prob max.', metavar='FLOAT',                 type=click.FloatRange(), default=0.9, show_default=True)
@click.option('--w_max',         help='Guidance level max. Code\'s w == Paper\'s w-1.', metavar='FLOAT', type=click.FloatRange(), default=14, show_default=True)
@click.option('--w_min',         help='Guidance level min. Code\'s w == Paper\'s w-1.', metavar='FLOAT', type=click.FloatRange(), default=0, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--enable_amp',    help='Enable torch.cuda.amp.GradScaler for mixed precision training, \
                                       NOTE overwritting loss_scale set by --ls', metavar='BOOL',   type=bool, default=False, show_default=True)
@click.option('--enable_tf32',   help='Enable tf32 for A100/H100 training speed', metavar='BOOL',   type=bool, default=False, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True) 
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True) 

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.FloatRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--ckpt',          help='How often to save latest checkpoints', metavar='TICKS',      type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from training state', metavar='SAFETENSORS',          type=str)
@click.option('--resume_dir',    help='Resume training state from a directory', metavar='DIR',      type=str)
@click.option('--resume-tick',   help='Number of tick from training state', metavar='INT',          type=int)
@click.option('--fixed_dir',     help='Log progress from (existing) fixed directory.', is_flag=True)
@click.option('-n', '--dry_run', help='Print training options and exit',                            is_flag=True)

# Evaluation
@click.option('--mid_t',         help='Sampler steps [default: 0.821]',                             multiple=True, default=[0.821]) #!?
@click.option('--metrics',       help='Comma-separated list or "none" [default: fid50k_full]',      type=CommaSeparatedList(), default='fid50k_full')
@click.option('--sample_every',  help='How often to sample imgs', metavar='TICKS',                  type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--eval_every',    help='How often to evaluate metrics', metavar='TICKS',             type=click.IntRange(min=1), default=50, show_default=True)

def main(**kwargs):
    """Train iGCT using the techniques described in the 
    "Beyond and Free from Diffusion: Invertible Guided Consistency Training".
    """

    opts = dnnlib.EasyDict(kwargs)
    #torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_start_method('forkserver', force=True)
    dist.init()
    
    # Initialize config dict. 
    c = dnnlib.EasyDict()
    
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    if opts.test_data:
        c.test_dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.test_data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict(P_mean=opts.mean, P_std=opts.std, q=opts.q, c=opts.c, k=opts.k, b=opts.b, 
                                    inv_P_mean=opts.inv_mean, inv_P_std=opts.inv_std, recon=opts.recon, recon_sch=opts.recon_sch,
                                    guide_t_low=opts.guide_t_low, guide_t_high=opts.guide_t_high, guide_p_pw=opts.guide_p_pw, guide_p_max=opts.guide_p_max, 
                                    w_min=opts.w_min, w_max=opts.w_max, 
                                    )
    c.optimizer_kwargs = dnnlib.EasyDict(class_name=f'torch.optim.{opts.optim}', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    if opts.test_data:
        try:
            test_dataset_obj = dnnlib.util.construct_class_by_name(**c.test_dataset_kwargs)
            c.test_dataset_kwargs.resolution = test_dataset_obj.resolution # be explicit about dataset resolution
            c.test_dataset_kwargs.max_size = len(test_dataset_obj) # be explicit about dataset size
            if opts.cond and not test_dataset_obj.has_labels:
                raise click.ClickException('--cond=True requires labels specified in dataset.json')
            del test_dataset_obj # conserve memory
        except IOError as err:
            raise click.ClickException(f'--test_data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ddpmpp-deep':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2], num_blocks=8)
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp-deep':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2], num_blocks=8)
    elif opts.arch == 'adm-small':
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,2,3])
    elif opts.arch == 'adm':
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    assert opts.precond == "igct", 'Unrecognized Precond & Loss!'
    c.network_kwargs.class_name = 'training.networks.IGCTPrecond'
    c.loss_kwargs.class_name = 'training.loss.IGCTLoss'

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    assert opts.guidance
    c.guidance_kwargs = dnnlib.EasyDict(class_name='training.guidance.GuidancePipe', w_embed_dim=opts.w_embed_dim)
    c.network_kwargs.w_embed_dim = opts.w_embed_dim
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000) if opts.ema is not None else opts.ema
    c.ema_beta = opts.ema_beta
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu) 
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench, enable_tf32=opts.enable_tf32, enable_amp=opts.enable_amp)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump, ckpt_ticks=opts.ckpt, double_ticks=opts.double)
    c.update(mid_t=opts.mid_t, metrics=opts.metrics, sample_ticks=opts.sample_every, eval_ticks=opts.eval_every)
    c.update(recon_every=opts.recon_every)
    c.update(im64_subg_dir=opts.im64_subg_dir)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None

    # Resume training state from directory.
    elif opts.resume_dir is not None and os.path.isdir(opts.resume_dir) and os.path.exists(opts.resume_dir):
        safetensors_files = [x for x in os.listdir(opts.resume_dir) if x.endswith('.safetensors')]
        if len(safetensors_files)>0:
            sft_ids = [re.fullmatch(r'training-state-(\d+).safetensors', x) for x in safetensors_files]
            sft_ids = [int(x.group().split('.')[0].split('-')[-1]) for x in sft_ids if x is not None]
            newest_sft_id = max(sft_ids, default=-1)
            newest_safetensors = os.path.join(opts.resume_dir, f'training-state-{newest_sft_id:06d}.safetensors')

            match = re.fullmatch(r'training-state-(\d+).safetensors', os.path.basename(newest_safetensors))
            c.resume_pkl = os.path.join(opts.resume_dir, f'network-snapshot-{match.group(1)}.pkl')
            c.resume_tick = int(match.group(1)) if opts.resume_tick is None else opts.resume_tick
            c.resume_state_dump = newest_safetensors
        else:
            match = re.fullmatch(r'training-state-(\d+|latest).safetensors', os.path.basename(opts.resume))
            if not match or not os.path.isfile(opts.resume):
                raise click.ClickException('--resume must point to training-state-*.safetensors from a previous training run')
            c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
            c.resume_tick = int(match.group(1)) if opts.resume_tick is None else opts.resume_tick
            c.resume_state_dump = opts.resume

    # Resume training state.
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+|latest).safetensors', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.safetensors from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_tick = int(match.group(1)) if opts.resume_tick is None else opts.resume_tick
        c.resume_state_dump = opts.resume
        
    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-{opts.optim:s}-{opts.lr:f}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{desc}') if opts.fixed_dir else os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
    
    # Convert run_dir to bytes for broadcasting
    if c.run_dir is None:
        run_dir_bytes = torch.ByteTensor(256).fill_(0).to('cuda')
    else:
        run_dir_bytes = torch.ByteTensor(list(c.run_dir.encode('utf-8'))).to('cuda')
        run_dir_bytes = torch.cat([run_dir_bytes, torch.ByteTensor(256 - len(run_dir_bytes)).fill_(0).to('cuda')])
    dist.broadcast(run_dir_bytes, src=0)
    c.run_dir = run_dir_bytes.cpu().numpy().tobytes().decode('utf-8').rstrip('\x00')

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Test dataset path:       {c.test_dataset_kwargs.path if opts.test_data else "None"}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0(f'Enabled amp:             {opts.enable_amp}')
    dist.print0(f'Enabled tf32:            {opts.enable_tf32}')
    dist.print0()

    # Dry run
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------