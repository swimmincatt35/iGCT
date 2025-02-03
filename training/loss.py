import math

import torch
import torch.nn as nn
from torch_utils import persistence
from torch_utils import distributed as dist


#----------------------------------------------------------------------------
# Loss function in "Beyond and Free from Diffusion: Invertible Guided Consistency Training" (iGCT)

@persistence.persistent_class
class IGCTLoss:
    def __init__(self, P_mean=-1.1, P_std=2.0, sigma_data=0.5, q=2, c=0.03, k=8.0, b=1.0, 
                 t_max=80, t_min=0.002, inv_P_mean=-1.1, inv_P_std=2.0, recon=1e-4, recon_sch=False,
                 guide_t_low=80, guide_t_high=14.3, guide_p_pw=2.0, guide_p_max=0.9, w_min=0, w_max=14, 
                 ):  

        self.P_mean = P_mean
        self.P_std = P_std
        self.inv_P_mean = inv_P_mean
        self.inv_P_std = inv_P_std
        self.rho = 7.0
        self.sigma_data = sigma_data
        
        self.t_to_r = self.t_to_r_sigmoid

        self.q = q
        self.stage = 0
        self.ratio = 0.
        self.k = k
        self.b = b
        self.c = c
        
        self.recon = recon
        self.recon_sch = recon_sch
        if recon_sch:
            dist.print0("Using hard coded reconstruction schedule for ImageNet64.")

        self.inv_t_r = self.inv_t_r_same
        self.t_max = t_max
        self.t_min = t_min

        self.guide_t_low = guide_t_low

        self.guide_t_high = guide_t_high
        self.guide_p_pw = guide_p_pw
        self.guide_p_max = guide_p_max

        self.w_min = w_min
        self.w_max = w_max 

        dist.print0(f'P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}, t_max: {self.t_max}')
        dist.print0(f'inv_P_mean: {self.inv_P_mean}, inv_P_std: {self.inv_P_std}, recon: {self.recon}')
        dist.print0(f'guide_t_low: {self.guide_t_low}, guide_t_high: {self.guide_t_high}, guide_p_pw: {self.guide_p_pw}, guide_p_max: {self.guide_p_max}')
        dist.print0(f'w_min: {self.w_min}, w_max: {self.w_max}')

    def update_schedule(self, stage):
        self.stage = stage
        self.ratio = 1 - 1 / self.q ** (stage)

    # Hard coded reconstruction schedule for ImageNet64
    def update_recon_schedule(self, ticks):
        if self.recon_sch:
            if ticks >= 0 and ticks < 1800:
                self.recon = 2e-5 
            elif ticks >= 1800 and ticks < 2000:
                self.recon = 4e-5
            else:
                self.recon = 6e-5

    def t_to_r_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)
    
    def inv_t_r_same(self,t,r):
        return torch.clamp(r, min=self.t_min), torch.clamp(t, min=self.t_min)
    
    def __call__(self, net, images, labels=None, guidance_pipe=None, do_recon_loss=True):
        inv_t, inv_loss = self.consistency_loss(net, images, labels, inversion=True, guidance_pipe=guidance_pipe)
        gen_t, gen_loss = self.consistency_loss(net, images, labels, inversion=False, guidance_pipe=guidance_pipe)
        loss = inv_loss + gen_loss

        if self.recon and do_recon_loss:
            rec_loss = self.reconstruction_loss(net, images, labels, guidance_pipe)
            loss += rec_loss * self.recon
        return loss, inv_t, inv_loss, gen_t, gen_loss
    
    def consistency_loss(self, net, images, labels, inversion, guidance_pipe):
        bs = images.shape[0]//2 if guidance_pipe else images.shape[0]
        rnd_normal = torch.randn([bs, 1, 1, 1], device=images.device)
        if inversion:
            t = torch.clamp((rnd_normal * self.inv_P_std + self.inv_P_mean).exp(), max=self.t_max)
            r = self.t_to_r(t)
            t,r = self.inv_t_r(t,r)  
        else:
            t = torch.clamp((rnd_normal * self.P_std + self.P_mean).exp(), max=self.t_max)  
            r = self.t_to_r(t)

        y_c = images[:bs] # images: images_c | images_d, labels: labels_c

        if guidance_pipe and not inversion:
            y_d = images[bs:]

            w = self.w_min + (self.w_max - self.w_min) * torch.rand(bs)
            w_embeds = guidance_pipe(w)
            w_embeds = w_embeds.to(images.device)
            w = w.to(images.device).view(bs, 1, 1, 1)

            power_prob = (t-self.guide_t_low) / (self.guide_t_high-self.guide_t_low) 
            power_prob = ((torch.clamp(power_prob, min=0, max=1)) ** self.guide_p_pw ) * self.guide_p_max
            guided_mask = torch.bernoulli(power_prob).bool()

            y = y_d * guided_mask + y_c * ~guided_mask
            
        else:
            w_embeds = None
            y = y_c

        # Shared noise direction
        eps   = torch.randn_like(y)
        eps_t = eps * t
        
        rng_state = torch.cuda.get_rng_state()
        D_yt = net(y + eps_t, t, labels, inversion=inversion, w_embeds=w_embeds)

        if inversion: # r > t
            eps_r = eps * r
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                D_yr = net(y + eps_r, r, labels, inversion=inversion, w_embeds=w_embeds) 

        else: # r >= 0
            if r.max() > 0:
                if guidance_pipe:
                    eps_r = (r-t) * (1+w) * (y-y_c)/t + eps * r  
                else:
                    eps_r = eps * r
                torch.cuda.set_rng_state(rng_state)
                with torch.no_grad():
                    D_yr = net(y + eps_r, r, labels, inversion=inversion, w_embeds=w_embeds) 
                mask = r > 0
                D_yr = torch.nan_to_num(D_yr)
                D_yr = mask * D_yr + (~mask) * y
            else:
                D_yr = y

        # L2 Loss
        loss = (D_yt - D_yr) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        # Huber Loss if needed
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        
        # Weighting fn 
        if inversion:
            return t.flatten(), loss * (self.sigma_data / self.t_max) * (r - t).flatten() 
        else:
            return t.flatten(), loss / (t - r).flatten()
    
    def reconstruction_loss(self, net, images, labels, guidance_pipe):
        bs = images.shape[0]//2 if guidance_pipe else images.shape[0]
        y = images[:bs]

        if guidance_pipe:
            w = torch.zeros(bs)
            w_embeds = guidance_pipe(w)
            w_embeds = w_embeds.to(images.device)
        else:
            w_embeds = None

        eps = torch.randn_like(y)
        y = y + eps*self.t_min
        recon_y, _ = net(y, None, src_labels=labels, tar_labels=labels, full_pipe=True, w_embeds=w_embeds) 

        # L2 Loss
        loss = (recon_y - y) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)

        # Huber Loss if needed
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        
        return loss

#----------------------------------------------------------------------------
# Reimplementation of baseline guided-CD in "Beyond and Free from Diffusion: Invertible Guided Consistency Training" (iGCT)
# Inspired by the paper "Consistency Models" https://arxiv.org/abs/2303.01469
# and "Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference" https://arxiv.org/abs/2310.04378

@persistence.persistent_class
class CDLoss:
    def __init__(self, sigma_data=0.5, t_max=80, t_min=0.002, w_min=0, w_max=14, N=18, rho=7, c=0.03):

        self.sigma_data = sigma_data
        self.t_max = t_max
        self.t_min = t_min
        self.rho = rho

        self.w_min = w_min
        self.w_max = w_max 
        self.N = N
        self.c = c

        dist.print0(f't_max: {self.t_max}, t_min: {self.t_min}, w_min: {self.w_min}, w_max: {self.w_max}, N: {self.N}, rho: {self.rho}, c: {self.c}')
    
    def sample_t_r(self, shape, device):
        indices = torch.randint(1, self.N, shape, device=device)
        t = ( self.t_max**(1/self.rho) + (indices-1)/(self.N-1) * (self.t_min**(1/self.rho) - self.t_max**(1/self.rho)) ) ** self.rho
        r = ( self.t_max**(1/self.rho) + indices/(self.N-1) * (self.t_min**(1/self.rho) - self.t_max**(1/self.rho)) ) ** self.rho
        return t, r

    def __call__(self, net, target_net, teacher_net, images, labels=None, guidance_pipe=None):
        bs = images.shape[0]
        y = images

        # Sample here
        t, r = self.sample_t_r(shape=(bs,1,1,1), device=images.device)

        if guidance_pipe:
            w = self.w_min + (self.w_max - self.w_min) * torch.rand(bs)
            w_embeds = guidance_pipe(w)
            w_embeds = w_embeds.to(images.device)
            w = w.to(images.device).view(bs, 1, 1, 1)

        else:
            w_embeds = None
        
        # Shared noise direction
        eps   = torch.randn_like(y)
        eps_t = eps * t
        y_t   = y + eps_t
            
        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        D_yt = net(y_t, t, labels, w_embeds=w_embeds) 

        @torch.no_grad()
        def heun_solver(samples, t, next_t, do_guidance=False):
            x = samples
            null_labels = torch.zeros_like(labels)

            denoiser = teacher_net(x, t, labels) # EDMPrecond
            d = (x - denoiser) / t
            if do_guidance:
                denoiser = teacher_net(x, t, null_labels)
                null_d = (x - denoiser) / t
                d = (1+w) * d - w * null_d
            samples = x + d * (next_t-t)
            
            denoiser = teacher_net(samples, next_t, labels)
            next_d = (samples - denoiser) / next_t
            if do_guidance:
                denoiser = teacher_net(samples, next_t, null_labels)
                next_null_d = (samples - denoiser) / next_t
                next_d = (1+w) * next_d - w * next_null_d

            samples = x + (d + next_d) * (next_t - t) / 2
            return samples
        
        if guidance_pipe:
            y_r = heun_solver(y_t, t, r, do_guidance=True)
        else:
            y_r = heun_solver(y_t, t, r)

        torch.cuda.set_rng_state(rng_state)
        with torch.no_grad():
            D_yr = target_net(y_r, r, labels, w_embeds=w_embeds) 
        
        # L2 Loss
        loss = (D_yt - D_yr) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        # Huber Loss if needed
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        return loss 