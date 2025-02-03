import copy
import numpy as np
import torch
import dnnlib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.nn.functional as F
import lpips


#----------------------------------------------------------------------------

def compute_recon_metrics(opts, metrics_list=["mse","psnr","ssim","lpips"]):
    dataset = dnnlib.util.construct_class_by_name(**opts.test_dataset_kwargs)

    device = opts.device
    num_items = len(dataset)
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=128, **data_loader_kwargs)
    
    def compute_psnr(img1, img2):
        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()
        return peak_signal_noise_ratio(img1, img2, data_range=2)

    def compute_ssim(img1, img2):
        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()
        if img1.ndim == 4:
            img1 = img1.transpose(0, 2, 3, 1)  # NHWC
            img2 = img2.transpose(0, 2, 3, 1)  # NHWC
        elif img1.ndim == 3:
            img1 = img1.transpose(1, 2, 0)  # HWC
            img2 = img2.transpose(1, 2, 0)  # HWC
        return structural_similarity(img1, img2, channel_axis=-1, data_range=2)

    lpips_model = lpips.LPIPS(net='alex').to(device)
    metrics_dict = {metric: [] for metric in metrics_list}

    for images, _labels in dataloader:
        images = images.to(device).to(torch.float32) / 127.5 - 1
        _labels = _labels.to(device)
        recon_images, _ = opts.reconstruction_fn(opts.G, images, _labels, **opts.G_kwargs)

        if "mse" in metrics_list:
            mse = F.mse_loss(recon_images, images).item()
            metrics_dict["mse"].append(mse)

        if "psnr" in metrics_list:
            psnr = compute_psnr(recon_images, images)
            metrics_dict["psnr"].append(psnr)
        
        if "ssim" in metrics_list:
            ssim = compute_ssim(recon_images, images)
            metrics_dict["ssim"].append(ssim)
        
        if "lpips" in metrics_list:
            lpips_value = lpips_model(images, recon_images).mean().item()
            metrics_dict["lpips"].append(lpips_value)
    
    for metric in metrics_list:
        metrics_dict[metric] = sum(metrics_dict[metric]) / len(metrics_dict[metric])
    
    return metrics_dict