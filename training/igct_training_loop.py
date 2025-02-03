import os
import time
import copy
import json
import pickle
import psutil
import functools
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from safetensors.torch import save_file, load_file
from metrics import metric_main

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

def assign_loss_quarter_bins(t, loss, t_max=80, t_min=0.002, inversion=False):
    inv = "inv " if inversion else ""

    q1 = t_min+(t_max-t_min)*0.25
    q2 = q1+(t_max-t_min)*0.25
    q3 = q2+(t_max-t_min)*0.25

    q1_loss = loss[t<=q1]
    q2_loss = loss[(t>q1)&(t<=q2)]
    q3_loss = loss[(t>q2)&(t<=q3)]
    q4_loss = loss[t>q3]

    return {f"Loss/loss {inv}q1": q1_loss, 
            f"Loss/loss {inv}q2": q2_loss,
            f"Loss/loss {inv}q3": q3_loss,
            f"Loss/loss {inv}q4": q4_loss
            }

#----------------------------------------------------------------------------

@torch.no_grad()
def generator_fn(
    net, latents, class_labels=None, w_embeds=None,
    t_max=80, mid_t=None
):
    # Time step discretization.
    mid_t = [] if mid_t is None else mid_t
    t_steps = torch.tensor([t_max]+list(mid_t), dtype=torch.float64, device=latents.device)

    # t_0 = T, t_N = 0
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Sampling steps 
    x = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x = net(x, t_cur, class_labels, w_embeds=w_embeds).to(torch.float64)
        x = torch.clamp(x, min=-1, max=1)
        if t_next > 0:
            x = x + t_next * torch.randn_like(x) 
    return x

#----------------------------------------------------------------------------

@torch.no_grad()
def reconstruction_fn(
    net, x, class_labels=None, w_embeds=None
):
    recon, inv = net(x, None, None, src_labels=class_labels, tar_labels=class_labels, full_pipe=True, w_embeds=w_embeds)
    return torch.clamp(recon, min=-1, max=1), inv

@torch.no_grad()
def editing_fn(
    net, x, src_labels, target_labels, w_embeds=None
):
    recon, inv = net(x, None, None, src_labels=src_labels, tar_labels=target_labels, full_pipe=True, w_embeds=w_embeds)
    return torch.clamp(recon, min=-1, max=1), inv

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    test_dataset_kwargs = {},       # Options for testing set. 
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    im64_subg_dir       = None,     # Options for imagenet64 editing metrics, directory to subgroups.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    guidance_kwargs     = None,     # Options for guidance pipeline, None = disable.
    visualize_gscales   = [0,6,12], # Guidance scales to visualize at evaluation.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_beta            = 0.9999,   # EMA decay rate. Overwritten by ema_halflife_kimg.
    ema_halflife_kimg   = None,     # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = None,     # EMA ramp-up coefficient, None = no rampup.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 500,      # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    ckpt_ticks          = 100,      # How often to save latest checkpoints, None = disable.
    sample_ticks        = 50,       # How often to sample images, None = disable.
    eval_ticks          = 500,      # How often to evaluate models, None = disable.
    double_ticks        = 500,      # How often to evaluate models, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_tick         = 0,        # Start from the given training progress.
    mid_t               = None,     # Intermediate t for few-step generation.
    metrics             = None,     # Metrics for evaluation. List.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark
    device              = torch.device('cuda'),
    recon_every         = 1,        # 10/2 added recon every for faster training
    enable_tf32         = False,    # Enable tf32 for A100/H100 GPUs?
    enable_amp          = False,    # Enable torch.cuda.amp.GradScaler,
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark

    # Enable these to speed up on A100 GPUs.
    torch.backends.cudnn.allow_tf32 = enable_tf32
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = enable_tf32

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    if guidance_kwargs:
        dataset_sampler = misc.PairedInfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    else:
        dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    data_loader = torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs)
    dataset_iterator = iter(data_loader)

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

    net.train().requires_grad_(True)
    
    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    guidance_pipe = dnnlib.util.construct_class_by_name(**guidance_kwargs)

    dist.print0(f'GradScaler enabled: {enable_amp} for mixed preicision training')
    if enable_amp:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-gradscaler
        # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
        dist.print0(f'Setting up GradScaler...')
        scaler = torch.cuda.amp.GradScaler()
        dist.print0(f'Loss scaling is overwritten when GradScaler is enabled')

    dist.print0('Setting up DDP...')
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False, find_unused_parameters=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Stats.
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            if network_kwargs.class_name == 'training.networks.IGCTPrecond':
                input_args = [images, None, None, labels, labels, False, False, True]
                misc.print_module_summary(net, input_args, max_nesting=2)
            else:
                misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        assert resume_tick >= 0
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = load_file(resume_state_dump)
        net_state_dict = {key.replace('net_', ''): data[key] for key in data if key.startswith('net_')}
        net.load_state_dict(net_state_dict)
        optimizer_state_dict = optimizer.state_dict()
        for key in optimizer_state_dict['state']:
            for sub_key in optimizer_state_dict['state'][key]:
                tensor_key = f'optimizer_state_{key}_{sub_key}'
                if tensor_key in data:
                    optimizer_state_dict['state'][key][sub_key] = data[tensor_key].to(device)
        optimizer.load_state_dict(optimizer_state_dict)

        if enable_amp:
            # NOTE(aiihn): Although not loading the state_dict of the GradScaler works well, 
            # loading it can improve reproducibility.
            dist.print0(f'Loading GradScaler state from "{resume_state_dump}"...')
            scaler_state_dict = scaler.state_dict()
            for key in scaler_state_dict:
                tensor_key = f'gradscaler_state_{key}'
                if tensor_key in data:
                    if 'growth' in tensor_key:
                        scaler_state_dict[key] = int(data[tensor_key])
                    else:
                        scaler_state_dict[key] = float(data[tensor_key])
            scaler.load_state_dict(scaler_state_dict)
        
        del data 

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    visualize_gscales = torch.Tensor(visualize_gscales)
    
    if dist.get_rank() == 0:
        dist.print0('Exporting sample images...')

        grid_size, data_images, data_labels = setup_snapshot_image_grid(training_set=dataset_obj)
        save_image_grid(data_images, os.path.join(run_dir, 'data.png'), drange=[0,255], grid_size=grid_size)
        
        grid_z = torch.randn([data_labels.shape[0], ema.img_channels, ema.img_resolution, ema.img_resolution], device=device)
        grid_z = grid_z.split(batch_gpu)
        
        grid_c = torch.from_numpy(data_labels).to(device)
        grid_c = grid_c.split(batch_gpu)

        grid_data_images = torch.from_numpy(data_images).to(device)
        norm_grid_data_images = (grid_data_images.float() / 255.0) * 2 - 1
        norm_grid_data_images = norm_grid_data_images.split(batch_gpu)
        
        for w in visualize_gscales:
            w_embeds = guidance_pipe(w.view(1))
            w_embeds = w_embeds.to(device)
            images = [generator_fn(ema, z, c, w_embeds).cpu() for z, c in zip(grid_z, grid_c)]
            images = torch.cat(images).numpy()

            subdir_path = os.path.join(run_dir, f"w={int(w)}")
            os.makedirs(subdir_path, exist_ok=True)
            save_image_grid(images, os.path.join(subdir_path, 'model_init.png'), drange=[-1,1], grid_size=grid_size)

            if w == 0:
                recon_images, inv_images = zip(*[reconstruction_fn(ema, norm_data_images, c, w_embeds) for norm_data_images, c in zip(norm_grid_data_images, grid_c)])
                recon_images = [img.cpu() for img in recon_images]
                inv_images = [inv.cpu() for inv in inv_images]

                recon_images = torch.cat(recon_images).numpy()
                inv_images = torch.cat(inv_images).numpy()

                save_image_grid(recon_images, os.path.join(subdir_path, 'recon_init.png'), drange=[-1,1], grid_size=grid_size)
                save_image_grid(np.clip(inv_images/80.0, a_min=-3, a_max=3), os.path.join(subdir_path, 'inv_init.png'), [-3, 3], grid_size)

                del recon_images            
                del inv_images

            del images
    
    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_tick * kimg_per_tick * 1000
    cur_tick = resume_tick
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg / 1000, total_kimg)
    stats_jsonl = None

    # Prepare for the mapping fn p(r|t).
    dist.print0(f'Reduce dt every {double_ticks} ticks.')
    
    def update_scheduler(loss_fn):
        loss_fn.update_schedule(stage)
        dist.print0(f'Update scheduler at/before {cur_tick} ticks, {cur_nimg / 1e3} kimg, ratio {loss_fn.ratio}')
    
    def update_recon_scheduler(loss_fn): # !?
        loss_fn.update_recon_schedule(cur_tick)
        
    stage = cur_tick // double_ticks
    update_scheduler(loss_fn)
    update_recon_scheduler(loss_fn) # !?

    iters = 0 
    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):                

                (images_c, labels_c), (images_d, labels_d) = next(dataset_iterator)
                images = torch.cat((images_c, images_d), dim=0)
                labels = labels_c

                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)

                do_recon_loss = True if iters % recon_every == 0 else False

                loss, inv_t, inv_loss, gen_t, gen_loss = loss_fn(net=ddp, images=images, labels=labels, guidance_pipe=guidance_pipe, do_recon_loss=do_recon_loss)
                for k,v in assign_loss_quarter_bins(gen_t, gen_loss, inversion=False).items():
                    training_stats.report(k, v)
                for k,v in assign_loss_quarter_bins(inv_t, inv_loss, inversion=True).items():
                    training_stats.report(k, v)
                    
                training_stats.report('Loss/loss', loss)
                if enable_amp:
                    scaler.scale(loss.mean()).backward()
                else:
                    loss.mul(loss_scaling).mean().backward()

        # Update weights.
        if enable_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Update EMA.
        if ema_halflife_kimg is not None:
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Update iterations.
        iters += 1

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"loss {training_stats.default_collector['Loss/loss']:<9.5f}"]

        if network_kwargs.class_name == 'training.networks.IGCTPrecond':
            fields += [f"q1 loss (inv) {training_stats.default_collector['Loss/loss inv q1']:<9.5f}"]
            fields += [f"q1 loss (gen) {training_stats.default_collector['Loss/loss q1']:<9.5f}"]
            fields += [f"q2 loss (inv) {training_stats.default_collector['Loss/loss inv q2']:<9.5f}"]
            fields += [f"q2 loss (gen) {training_stats.default_collector['Loss/loss q2']:<9.5f}"]
            fields += [f"q3 loss (inv) {training_stats.default_collector['Loss/loss inv q3']:<9.5f}"]
            fields += [f"q3 loss (gen) {training_stats.default_collector['Loss/loss q3']:<9.5f}"]
            fields += [f"q4 loss (inv) {training_stats.default_collector['Loss/loss inv q4']:<9.5f}"]
            fields += [f"q4 loss (gen) {training_stats.default_collector['Loss/loss q4']:<9.5f}"]
        
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0) and cur_tick != 0:
            data = dict(ema=ema, loss_fn=loss_fn, guidance_pipe=guidance_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_tick:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            save_dict = {}

            net_state_dict = net.state_dict()
            for key, value in net_state_dict.items():
                if isinstance(value, torch.Tensor):
                    save_dict[f'net_{key}'] = value

            optimizer_state_dict = optimizer.state_dict()
            for key, value in optimizer_state_dict['state'].items():
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        save_dict[f'optimizer_state_{key}_{sub_key}'] = sub_value
            
            if enable_amp:
                scaler_state_dict = scaler.state_dict()
                for key, value in scaler_state_dict.items():
                    save_dict[f'gradscaler_state_{key}'] = torch.tensor(value)     

            save_file(save_dict, os.path.join(run_dir, f'training-state-{cur_tick:06d}.safetensors'))
        
        # Save latest network snapshot.
        if (ckpt_ticks is not None) and (done or cur_tick % ckpt_ticks == 0) and cur_tick != 0:
            dist.print0(f'Save the latest checkpoint at {cur_tick:06d} img...')
            data = dict(ema=ema, loss_fn=loss_fn, guidance_pipe=guidance_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value 
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-latest.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data 

            # Save latest training state.
            if dist.get_rank() == 0:
                save_dict = {}

                net_state_dict = net.state_dict()
                for key, value in net_state_dict.items():
                    if isinstance(value, torch.Tensor):
                        save_dict[f'net_{key}'] = value
                
                optimizer_state_dict = optimizer.state_dict()
                for key, value in optimizer_state_dict['state'].items():
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            save_dict[f'optimizer_state_{key}_{sub_key}'] = sub_value
                
                if enable_amp:
                    scaler_state_dict = scaler.state_dict()
                    for key, value in scaler_state_dict.items():
                        save_dict[f'gradscaler_state_{key}'] = torch.tensor(value)
            
                save_file(save_dict, os.path.join(run_dir, f'training-state-latest.safetensors'))


        # Sample/visualize images.
        if (sample_ticks is not None) and (done or cur_tick % sample_ticks == 0) and dist.get_rank() == 0:
            dist.print0('Exporting sample images...')

            for w in visualize_gscales:
                w_embeds = guidance_pipe(w.view(1))
                w_embeds = w_embeds.to(device)
                images = [generator_fn(ema, z, c, w_embeds).cpu() for z, c in zip(grid_z, grid_c)]
                images = torch.cat(images).numpy()

                subdir_path = os.path.join(run_dir, f"w={int(w)}")
                os.makedirs(subdir_path, exist_ok=True)
                save_image_grid(images, os.path.join(subdir_path, f'{cur_tick:06d}.png'), drange=[-1,1], grid_size=grid_size)
                del images

                if w == 0:
                    recon_images, inv_images = zip(*[reconstruction_fn(ema, norm_data_images, c, w_embeds) for norm_data_images, c in zip(norm_grid_data_images, grid_c)])
                    recon_images = [img.cpu() for img in recon_images]
                    inv_images = [inv.cpu() for inv in inv_images]

                    recon_images = torch.cat(recon_images).numpy()
                    inv_images = torch.cat(inv_images).numpy()

                    save_image_grid(recon_images, os.path.join(subdir_path, f'recon_{cur_tick:06d}.png'), drange=[-1,1], grid_size=grid_size)
                    save_image_grid(np.clip(inv_images/80.0, a_min=-3, a_max=3), os.path.join(subdir_path, f'inv_{cur_tick:06d}.png'), [-3, 3], grid_size)
                    del recon_images            
                    del inv_images

 
        # Evaluation and editing visualization.
        if (eval_ticks is not None) and (done or cur_tick % eval_ticks == 0):
            
            # Cifar10, 10x10 classes cross editing visualization.
            if dataset_obj.name=="cifar10-32x32":
                if dist.get_rank() == 0 and dataset_obj.has_labels:
                    editing_run_dir = os.path.join(run_dir, "editing_cifar10")
                    if not os.path.exists(editing_run_dir):
                        os.makedirs(editing_run_dir)
                        dist.print0(f"Directory '{editing_run_dir}' created.")
                    
                    dist.print0('Exporting sample images (editing)...')
                    num_classes = dataset_obj.label_dim
                    one_hot_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int).to(device)
                    
                    for w in visualize_gscales:
                        w_embeds = guidance_pipe(w.view(1))
                        w_embeds = w_embeds.to(device)

                        subdir_path = os.path.join(editing_run_dir, f"w={int(w)}")
                        os.makedirs(subdir_path, exist_ok=True)

                        for i in range(num_classes):
                            one_hot_matrix[i,i] = 1
                            target_label = one_hot_matrix[i].to(device)
                            edit_images, _ = zip(*[editing_fn(ema, norm_data_images, c, target_label.repeat(c.shape[0],1), w_embeds) for norm_data_images, c in zip(norm_grid_data_images, grid_c)])
                            edit_images = [img.cpu() for img in edit_images]
                            edit_images = torch.cat(edit_images).numpy()
                            save_image_grid(edit_images, os.path.join(subdir_path, f'edit_{cur_tick:06d}_{i}.png'), drange=[-1,1], grid_size=grid_size)

                    del edit_images 
                    del one_hot_matrix

            # ImageNet64, subgroup editing visualization directory.
            if "im64_edit_full" in metrics:
                im64_visualize_dir = os.path.join(run_dir,"editing_im64",f"tick={cur_tick:06d}")
                if dist.get_rank() == 0:
                    if not os.path.exists(im64_visualize_dir):
                        os.makedirs(im64_visualize_dir)
            else:
                im64_visualize_dir = None
            torch.distributed.barrier()

            dist.print0('Evaluating models...')
            for metric in metrics:

                dist.print0(f'  Evaluating metric: {metric}')       
                eval_gscales = torch.Tensor([0]) if metric == "recon_full" else visualize_gscales 

                for w in eval_gscales:
                    w_embeds = guidance_pipe(w.view(1))
                    w_embeds = w_embeds.to(device)
                    
                    result_dict = metric_main.calc_metric(
                        metric=metric, 
                        generator_fn=generator_fn, 
                        reconstruction_fn=reconstruction_fn,
                        editing_fn=editing_fn,
                        G=ema, 
                        G_kwargs={"w_embeds":w_embeds},
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
                        metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'network-snapshot-w={w}-{cur_tick:06d}.pkl')                        

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg / 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break
        
        # Update Scheduler
        new_stage = (cur_tick-1) // double_ticks
        if new_stage > stage:
            stage = new_stage
            update_scheduler(loss_fn)
        update_recon_scheduler(loss_fn)

    # Few-step Evaluation.
    few_step_fn = functools.partial(generator_fn, mid_t=mid_t)
    
    if dist.get_rank() == 0:
        dist.print0('Exporting final sample images...')
        images = [few_step_fn(ema, z, c).cpu() for z, c in zip(grid_z, grid_c)]
        images = torch.cat(images).numpy()
        save_image_grid(images, os.path.join(run_dir, 'final.png'), drange=[-1,1], grid_size=grid_size)
        del images

    dist.print0('Evaluating few-step generation...')
    for _ in range(3):
        for metric in ["fid50k_full","pr50k3_full"]:
            for w in visualize_gscales:
                w_embeds = guidance_pipe(w.view(1))
                w_embeds = w_embeds.to(device)                
                result_dict = metric_main.calc_metric(metric=metric, 
                                                  generator_fn=few_step_fn, 
                                                  G=ema, 
                                                  G_kwargs={"w_embeds":w_embeds},
                                                  dataset_kwargs=dataset_kwargs, 
                                                  num_gpus=dist.get_world_size(), 
                                                  rank=dist.get_rank(), 
                                                  device=device
                                                  )
                if dist.get_rank() == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'network-snapshot-fewsteps-w-{w}-latest.pkl')                        
   
    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------