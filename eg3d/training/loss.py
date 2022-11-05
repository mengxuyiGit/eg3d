# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from training.chamfer_loss import ChamferLoss

from ipdb import set_trace as st
from training.volume import VolumeGenerator

import clip
import PIL
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from training.training_loop import save_fetched_data
    
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased',
                    use_perception=False, perception_reg=1, 
                    use_l2=False, l2_reg=1, use_l1=False, l1_reg=1,
                    use_chamfer=False, chamfer_reg=1,
                    use_patch=False, patch_reg=1,
                    discriminator_condition_on_real=False, drop_pixel_ratio=0.8):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)
        self.use_perception  = use_perception
        self.perception_reg = perception_reg
        if self.use_perception:
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # self.perception_reg = 1 # ViT will give larger loss
            # model, preprocess = clip.load("RN50", device=device)
            self.clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device) # maybe too large
            n_px = 224
            self.my_process = Compose([
                    Resize(n_px, interpolation=InterpolationMode.BICUBIC),
                    # CenterCrop(n_px), # unnecessary
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])

            for p in self.clip_model.parameters():
                p.requires_grad=False
                
        self.use_l2 = use_l2
        self.l2_reg = l2_reg

        self.use_l1 = use_l1
        if self.use_l1:
            self.l1_reg = l1_reg
            self.l1 = torch.nn.L1Loss(reduction='none')

        self.use_chamfer = use_chamfer
        if self.use_chamfer:
            self.chamfer_reg = chamfer_reg
            self.chamfer_loss = ChamferLoss()

        self.use_patchD = use_patch
        self.patchD_reg = patch_reg
        self.discriminator_condition_on_real = discriminator_condition_on_real
        self.drop_pixel_ratio  = drop_pixel_ratio

        if self.discriminator_condition_on_real:
            # assert not (self.use_l2 or self.use_l1 or self.use_chamfer or self.use_perception)
            # assert not (self.use_l2 or self.use_chamfer or self.use_perception)
            assert not (self.use_chamfer or self.use_perception)
            assert not (self.use_l2 and self.use_l1)


    def run_G(self, z, c, pc, swapping_prob, neural_rendering_resolution, update_emas=False):
        
        ws = None
        if isinstance(self.G, VolumeGenerator):
            gen_output = self.G.synthesis(ws, c, pc, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        else:
            gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        return gen_output, ws

    def cal_l2_loss(self, gen_img, real_img):

        gen_image_features = gen_img['image']
        real_image_features = real_img['image']
        
        # mse = torch.nn.MSELoss(reduction='none')
        # loss = mse(gen_image_features, real_image_features)
                        
        return (gen_image_features-real_image_features)**2
    
    def cal_l1_loss(self, gen_img, real_img):

        gen_image_features = gen_img['image']
        real_image_features = real_img['image']
        
        loss = self.l1(gen_image_features, real_image_features)
                        
        return loss
        

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_gt, gen_c, gen_pc, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']

        # phase_real_img.to(device).to(torch.float32) / 127.5 - 1
        
        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        neural_rendering_resolution=128 # hardcoded for debug

        real_img = {'image': real_img}
        gen_gt_img = {'image': gen_gt}

        if phase in ['Gmain', 'Gboth']:

            with torch.autograd.profiler.record_function('Gmain_forward'):
               
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_pc, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                

                # # L1 loss on the whole gen image
                # if self.use_l1:
                #     l1_loss = self.cal_l1_loss(gen_img=gen_img, real_img=gen_gt_img)
                #     # l1_loss = torch.mean(l1_loss.flatten(1), -1, True) * self.l1_reg
                #     l1_loss = torch.mean(l1_loss) * self.l1_reg
                #     loss_Gmain = l1_loss
                #     print(f"---------loss_l1\t\t(x{self.l1_reg}): {(l1_loss).sum().item()}-------------")

                #     training_stats.report('Loss/G/l1_loss_whole', l1_loss)

                # L2 loss on the whole gen image
                if self.use_l2:
                    l2_loss = self.cal_l2_loss(gen_img=gen_img, real_img=gen_gt_img)
                    l2_loss = torch.mean(l2_loss) * self.l2_reg

                    loss_Gmain = l2_loss
                    # print(f"---------loss_l2\t\t(x{self.l2_reg}): {(l2_loss).sum().item()}-------------")

                    training_stats.report('Loss/G/l2_loss_whole', l2_loss)
                    
                with torch.autograd.profiler.record_function('Gmain_backward'):
                    loss_Gmain.mean().mul(gain).backward()
                    # for name, param in self.G.backbone.synthesis.synthesis_unet3d.named_parameters():
         
                    #     if param.requires_grad:
                    #         print(name, param)

    
#----------------------------------------------------------------------------
