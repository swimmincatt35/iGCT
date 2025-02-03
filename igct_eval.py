import os
import re
import click
import torch
import dnnlib
import functools
from torch_utils import distributed as dist
from metrics import metric_main
from torch_utils import misc
import numpy as np
import pickle
import PIL.Image
from tqdm import tqdm


import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.


#----------------------------------------------------------------------------
# Step functions for discrete guided-cd and discrete cfg-edm

steps = 18
t_min = 0.002
t_max = 80.0
r = 7.0
indices = torch.linspace(0,steps-1,steps)
discrete_t = ( t_min**(1/r) + (indices)/(steps-1) * (t_max**(1/r) - t_min**(1/r)) ) ** r

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0, width=None, height=None):
    rnd = np.random.RandomState(random_seed)
    gw = width if width else np.clip(7680 // training_set.image_shape[2], 7, 16)
    gh = height if height else np.clip(4320 // training_set.image_shape[1], 4, 16)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        total_classes = len(label_order)
        for y in range(gh):
            label = label_order[y%len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)
   
#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size # 16, 288
    _N, C, H, W = img.shape
    assert C in [1, 3]

    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

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

@torch.no_grad()
def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    w_scale=0, inverse=False,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    if inverse:
        t_steps = t_steps.flip(dims=[0])
        x_next = latents
    else:
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        x_next = latents.to(torch.float32) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float32)
        d_cur = (x_hat - denoised) / t_hat

        # Apply cfg correction 10/10.
        if w_scale:
            null_class_labels = torch.zeros_like(class_labels)
            null_denoised = net(x_hat, t_hat, null_class_labels).to(torch.float32)
            d_null_cur = (x_hat - null_denoised) / t_hat
            d_cur = (1+w_scale) * d_cur - w_scale * d_null_cur
        
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float32)
            d_prime = (x_next - denoised) / t_next

            # Apply cfg correction 10/10.
            if w_scale:
                null_class_labels = torch.zeros_like(class_labels)
                null_denoised = net(x_next, t_next, null_class_labels).to(torch.float32)
                d_null_prime = (x_next - null_denoised) / t_next
                d_prime = (1+w_scale) * d_prime - w_scale * d_null_prime

            d_cur = 0.5 * d_cur + 0.5 * d_prime
            x_next = x_hat + (t_next - t_hat) * d_cur

    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def cm_sampler(
    net, latents, class_labels=None, w_embeds=None,
    t_max=80, mid_t=None 
):
    # Time step discretization.
    mid_t = [] if mid_t is None else mid_t
    t_steps = torch.tensor([t_max]+list(mid_t), dtype=torch.float32, device=latents.device)

    # t_0 = T, t_N = 0
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Sampling steps 
    x = latents.to(torch.float32) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x = net(x, t_cur, class_labels, w_embeds=w_embeds).to(torch.float32)
        x = torch.clamp(x, min=-1, max=1)
        if t_next > 0:
            x = x + t_next * torch.randn_like(x) 
    return x

#----------------------------------------------------------------------------

@torch.no_grad()
def igct_reconstruction(
    net, x, class_labels=None, w_embeds=None
):
    recon, inv = net(x, None, None, src_labels=class_labels, tar_labels=class_labels, full_pipe=True, w_embeds=w_embeds)
    return torch.clamp(recon, min=-1, max=1), inv

#----------------------------------------------------------------------------

@torch.no_grad()
def igct_editing(
    net, x, src_labels, target_labels, w_embeds=None
):
    recon, inv = net(x, None, None, src_labels=src_labels, tar_labels=target_labels, full_pipe=True, w_embeds=w_embeds)
    return torch.clamp(recon, min=-1, max=1), inv

#----------------------------------------------------------------------------

@torch.no_grad()
def edm_reconstruction(
    net, x, class_labels=None, w_scale=0
):
    inv = edm_sampler(net, x, class_labels=class_labels, inverse=True, w_scale=0)
    inv = inv/80.0
    recon = edm_sampler(net, inv, class_labels=class_labels, w_scale=w_scale)
    return torch.clamp(recon, min=-1, max=1), inv*80

@torch.no_grad()
def edm_editing(
    net, x, src_labels, target_labels, w_scale=0
):
    inv = edm_sampler(net, x, class_labels=src_labels, inverse=True, w_scale=0)
    inv = inv/80.0
    recon = edm_sampler(net, inv, class_labels=target_labels, w_scale=w_scale)
    return torch.clamp(recon, min=-1, max=1), inv*80

#----------------------------------------------------------------------------


@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--test_data',     help='Path to the testing dataset', metavar='ZIP|DIR',             type=str, default=None)
@click.option('--im64_subg_dir', help='Directory for im64 subgroups', metavar='DIR',                type=str, default=None)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture',                                       type=click.Choice(['ncsnpp','ddpmpp','ncsnpp-deep','ddpmpp-deep','adm-small','adm']), default='ncsnpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='igct|edm|cd',     type=click.Choice(['igct','edm','cd']), default='igct', show_default=True)

# Hyperparameters.
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)
@click.option('--w_embed_dim',   help='Dimension for w-embed ', metavar='INT',                      type=click.IntRange(min=1), default=256, show_default=True)

@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True) 
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True) 

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--net_pkl',       help='Network weights', metavar='PKL',                             type=str)
@click.option('-n', '--dry_run', help='Print training options and exit',                            is_flag=True)

# Evaluation
@click.option('--metrics',       help='Comma-separated list or "none" [default: fid50k_full]',      type=CommaSeparatedList(), default='fid50k_full')
@click.option('--few_step',      help='Few-step (2-step) generation.', metavar='BOOL',              type=bool, default=False, show_default=True) 


def main(**kwargs):
    """Evaluate iGCT from the paper
    "Beyond and Free from Diffusion: Invertible Guided Consistency Training".
    """

    device=torch.device('cuda')
    opts = dnnlib.EasyDict(kwargs)
    #torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_start_method('forkserver', force=True)
    dist.init()
    
    # Dataset.
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    if opts.test_data:
        test_dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.test_data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    
    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        dataset_name = dataset_obj.name
        dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    if opts.test_data:
        try:
            test_dataset_obj = dnnlib.util.construct_class_by_name(**test_dataset_kwargs)
            test_dataset_kwargs.resolution = test_dataset_obj.resolution # be explicit about dataset resolution
            test_dataset_kwargs.max_size = len(test_dataset_obj) # be explicit about dataset size
            if opts.cond and not test_dataset_obj.has_labels:
                raise click.ClickException('--cond=True requires labels specified in dataset.json')
            del test_dataset_obj # conserve memory
        except IOError as err:
            raise click.ClickException(f'--test_data: {err}')

    # Network architecture.
    network_kwargs = dnnlib.EasyDict()
    if opts.arch == 'ddpmpp':
        network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ddpmpp-deep':
        network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2], num_blocks=8)
    elif opts.arch == 'ncsnpp':
        network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp-deep':
        network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2], num_blocks=8)
    elif opts.arch == 'adm-small':
        network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,2,3])
    elif opts.arch == 'adm':
        network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    if opts.precond == "igct":
        network_kwargs.class_name = 'training.networks.IGCTPrecond'
    elif opts.precond == "edm":
        network_kwargs.class_name = 'training.networks.EDMPrecond'
    elif opts.precond == 'cd':
        network_kwargs.class_name = 'training.networks.CDPrecond'
    else:
        raise ValueError('Unrecognized Precond & Loss!')

    # Network options.
    if opts.cbase is not None:
        network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        network_kwargs.channel_mult = opts.cres
    if opts.precond != "edm": # guidance.GuidancePipe for cm's
        network_kwargs.w_embed_dim = opts.w_embed_dim
        guidance_kwargs = dnnlib.EasyDict(class_name='training.guidance.GuidancePipe', w_embed_dim=opts.w_embed_dim)
        guidance_pipe = dnnlib.util.construct_class_by_name(**guidance_kwargs) 
    else:
        guidance_pipe = None 

    # Random seed.
    if opts.seed is not None:
        seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        seed = int(seed)

    # Description string.
    cond_str = 'cond' if dataset_kwargs.use_labels else 'uncond'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{opts.batch:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        run_dir = None
    elif opts.nosubdir:
        run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(run_dir)
    
    # Convert run_dir to bytes for broadcasting
    if run_dir is None:
        run_dir_bytes = torch.ByteTensor(256).fill_(0).to('cuda')
    else:
        run_dir_bytes = torch.ByteTensor(list(run_dir.encode('utf-8'))).to('cuda')
        run_dir_bytes = torch.cat([run_dir_bytes, torch.ByteTensor(256 - len(run_dir_bytes)).fill_(0).to('cuda')])
    dist.broadcast(run_dir_bytes, src=0)
    run_dir = run_dir_bytes.cpu().numpy().tobytes().decode('utf-8').rstrip('\x00')

    # Other options.
    batch_size=opts.batch 
    net_pkl = opts.net_pkl
    metrics=opts.metrics
    im64_subg_dir=opts.im64_subg_dir

    # Print eval options.
    dist.print0()
    dist.print0('Eval options:')
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Dataset path:            {dataset_kwargs.path}')
    dist.print0(f'Test dataset path:       {test_dataset_kwargs.path if opts.test_data else "None"}')
    dist.print0(f'Class-conditional:       {dataset_kwargs.use_labels}')
    dist.print0(f'Network pkl path:        {net_pkl}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {batch_size}')
    dist.print0(f'Few step evaluation:     {opts.few_step}')
    dist.print0()

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(run_dir, exist_ok=True)
        dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Initialize.
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = True

    # Batch size per GPU.
    batch_gpu = batch_size // dist.get_world_size()

    # Load dataset.
    dist.print0('Loading training dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.to(device)
    torch.distributed.barrier() 

    # Broadcast network to other ranks.
    for param in net.parameters():
        dist.broadcast(param.data, src=0)
    for buffer in net.buffers():
        dist.broadcast(buffer.data, src=0)
    net.eval().requires_grad_(False)

    # Loading networks weights from pkl.
    dist.print0(f'Loading network weights from "{net_pkl}"...')
    if dist.get_rank() != 0:
        torch.distributed.barrier() # rank 0 goes first
    with dnnlib.util.open_url(net_pkl, verbose=(dist.get_rank() == 0)) as f:
        data = pickle.load(f)
    if dist.get_rank() == 0:
        torch.distributed.barrier() # other ranks follow
    misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
    del data # conserve memory

    if opts.precond == "edm":
        generator_fn = edm_sampler 
    else:
        if opts.few_step:
            if opts.precond == "cd":
                generator_fn = functools.partial(cm_sampler, mid_t=[discrete_t[7]])
            else:
                generator_fn = functools.partial(cm_sampler, mid_t=[0.821])
        else:
            generator_fn = cm_sampler
    editing_fn = edm_editing if opts.precond == "edm" else igct_editing
    reconstruction_fn = edm_reconstruction if opts.precond == "edm" else igct_reconstruction

    visualize_gscales = torch.Tensor([0,6,12])

    # Visualization.
    if dist.get_rank() == 0:
        dist.print0('Exporting sample images...')

        grid_size, data_images, data_labels = setup_snapshot_image_grid(training_set=dataset_obj)
        save_image_grid(data_images, os.path.join(run_dir, 'data.png'), drange=[0,255], grid_size=grid_size)
        
        grid_z = torch.randn([data_labels.shape[0], net.img_channels, net.img_resolution, net.img_resolution], device=device)
        grid_z = grid_z.split(batch_gpu)

        grid_c = torch.from_numpy(data_labels).to(device)
        grid_c = grid_c.split(batch_gpu)

        grid_data_images = torch.from_numpy(data_images).to(device)
        norm_grid_data_images = (grid_data_images.float() / 255.0) * 2 - 1
        norm_grid_data_images = norm_grid_data_images.split(batch_gpu)

        # Guidance visualization.
        for w in visualize_gscales:
            if guidance_pipe != None:
                w_embeds = guidance_pipe(w.view(1))
                w_embeds = w_embeds.to(device)
                images = [generator_fn(net, z, c, w_embeds).cpu() for z, c in zip(grid_z, grid_c)]
            else:
                images = [generator_fn(net, z, class_labels=c, w_scale=w).cpu() for z, c in zip(grid_z, grid_c)]
            images = torch.cat(images).numpy()
            subdir_path = os.path.join(run_dir, f"w={int(w)}")
            os.makedirs(subdir_path, exist_ok=True)
            save_image_grid(images, os.path.join(subdir_path, 'model_init.png'), drange=[-1,1], grid_size=grid_size)

            # Reconstruction visualization.
            if w == 0 and (opts.precond == "edm" or opts.precond == "igct"):
                if opts.precond == "edm":
                    recon_images, inv_images = zip(*[reconstruction_fn(net, norm_data_images, c) for norm_data_images, c in zip(norm_grid_data_images, grid_c)])
                else:
                    recon_images, inv_images = zip(*[reconstruction_fn(net, norm_data_images, c, w_embeds) for norm_data_images, c in zip(norm_grid_data_images, grid_c)])
                
                recon_images = [img.cpu() for img in recon_images]
                inv_images = [inv.cpu() for inv in inv_images]
                recon_images = torch.cat(recon_images).numpy()
                inv_images = torch.cat(inv_images).numpy()

                save_image_grid(recon_images, os.path.join(subdir_path, 'recon_init.png'), drange=[-1,1], grid_size=grid_size)
                save_image_grid(np.clip(inv_images/80.0, a_min=-3, a_max=3), os.path.join(subdir_path, 'inv_init.png'), [-3, 3], grid_size)

                del recon_images            
                del inv_images

            del images 

        # Cifar10 editing visualization.
        if dataset_obj.name=="cifar10-32x32" and (opts.precond == "edm" or opts.precond == "igct") :

            editing_run_dir = os.path.join(run_dir, "editing_cifar10")
            if not os.path.exists(editing_run_dir):
                os.makedirs(editing_run_dir)
                dist.print0(f"Directory '{editing_run_dir}' created.")
            
            dist.print0('Exporting sample images (editing)...')
            num_classes = dataset_obj.label_dim
            one_hot_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int).to(device)
            
            for w in visualize_gscales:
                if guidance_pipe != None:
                    w_embeds = guidance_pipe(w.view(1))
                    w_embeds = w_embeds.to(device)

                subdir_path = os.path.join(editing_run_dir, f"w={int(w)}")
                os.makedirs(subdir_path, exist_ok=True)

                for i in range(num_classes):
                    one_hot_matrix[i,i] = 1
                    target_label = one_hot_matrix[i].to(device)
                    
                    if opts.precond == "igct":
                        edit_images, _ = zip(*[editing_fn(net, norm_data_images, c, target_label.repeat(c.shape[0],1), w_embeds) for norm_data_images, c in zip(norm_grid_data_images, grid_c)])
                    else:
                        edit_images, _ = zip(*[editing_fn(net, norm_data_images, c, target_label.repeat(c.shape[0],1), w_scale=w) for norm_data_images, c in zip(norm_grid_data_images, grid_c)])

                    edit_images = [img.cpu() for img in edit_images]
                    edit_images = torch.cat(edit_images).numpy()
                    save_image_grid(edit_images, os.path.join(subdir_path, f'edit_init_{i}.png'), drange=[-1,1], grid_size=grid_size)

            del edit_images 
            del one_hot_matrix

    # ImageNet64, subgroup editing visualization directory.
    if "im64_edit_full" in metrics:
        im64_visualize_dir = os.path.join(run_dir,"editing_im64")
        if dist.get_rank() == 0:
            if not os.path.exists(im64_visualize_dir):
                os.makedirs(im64_visualize_dir)
    else:
        im64_visualize_dir = None
    torch.distributed.barrier()
    
    dist.print0('Evaluating models...')
    for metric in metrics:
        eval_gscales = torch.Tensor([0]) if metric == "recon_full" else visualize_gscales

        dist.print0(f'  Evaluating metric: {metric}')
        for w in tqdm(eval_gscales):
            
            if guidance_pipe:
                w_embeds = guidance_pipe(w.view(1))
                w_embeds = w_embeds.to(device)
                G_kwargs = {"w_embeds":w_embeds}
            else:
                w_embeds = None
                G_kwargs = {"w_scale":w}

            result_dict = metric_main.calc_metric(
                metric=metric, 
                generator_fn=generator_fn, 
                reconstruction_fn=reconstruction_fn,
                editing_fn=editing_fn,
                G=net, 
                G_kwargs=G_kwargs,
                dataset_kwargs=dataset_kwargs, 
                test_dataset_kwargs=test_dataset_kwargs, 
                im64_subg_dir=im64_subg_dir,
                im64_visualize_dir=im64_visualize_dir,
                w_scale=w,
                num_gpus=dist.get_world_size(), 
                rank=dist.get_rank(), 
                device=device
            )
            if dist.get_rank() == 0:
                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'{net_pkl}-w-{w}.pkl')                        

    dist.print0('Done')   

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------