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
                    discriminator_condition_on_real=False, 
                    drop_pixel_ratio=0.8,
                    discriminator_condition_on_projection=False, 
                    ):
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
        assert not self.use_patchD
        self.patchD_reg = patch_reg
        self.discriminator_condition_on_real = discriminator_condition_on_real
        self.drop_pixel_ratio  = drop_pixel_ratio
        self.discriminator_condition_on_projection = discriminator_condition_on_projection
        assert not (self.discriminator_condition_on_real and self.discriminator_condition_on_projection)

        if self.discriminator_condition_on_real or self.discriminator_condition_on_projection:
            # assert not (self.use_l2 or self.use_l1 or self.use_chamfer or self.use_perception)
            assert not (self.use_l2 or self.use_chamfer or self.use_perception)

    def run_G(self, z, c, pc, swapping_prob, neural_rendering_resolution, update_emas=False):
        
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        if self.G.z_from_pc:
            ws = self.G.mapping(None, c_gen_conditioning, pc, update_emas=update_emas)
        else:
            ws = self.G.mapping(z, c_gen_conditioning, None, update_emas=update_emas)
        
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        # st()
        if isinstance(self.G, VolumeGenerator):
            gen_output = self.G.synthesis(ws, c, pc, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        else:
            gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0,   update_emas=False):
        # st() # check img['image'].shape: ([B, 3, 128, 128]))
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)
        # st() # check img['image'].shape: ([B, 3, 128, 128]))
        logits = self.D(img, c, update_emas=update_emas)
        return logits
    
    def run_patchD(self, img, target, return_pred=False):
        # patch_loss = self.D.patchD_forward(img, target)
        patch_loss = self.D(img, target, return_pred)
        return patch_loss
    
    def cal_perception_loss(self, gen_img, real_img):
        
        # # convert to PIL 
        # transform = T.ToPILImage()

        # gen_img_PIL=[]
        # # currently all images are normalized within -1~1
        # for gi in gen_img['image']:
        #     # gi = (gi+1.)/2 # no difference
        #     gen_img_PIL.append(self.clip_preprocess(transform(gi)))
        # gen_img_processed = torch.tensor(np.stack(gen_img_PIL)).to(self.device)

        # real_img_PIL=[]
        # for ri in real_img['image']:
        #     # ri = (ri+1.)/2 # no difference
        #     real_img_PIL.append(self.clip_preprocess(transform(ri)))
        # real_img_processed = torch.tensor(np.stack(real_img_PIL)).to(self.device)

        ################## the above is discarded ####################

        gen_img_processed = self.my_process(gen_img['image'])
        real_img_processed = self.my_process(real_img['image'])
       
        gen_image_features = self.clip_model.encode_image(gen_img_processed).float()
        real_image_features = self.clip_model.encode_image(real_img_processed).float()
        
        mse = torch.nn.MSELoss(reduction='none')
        loss = mse(gen_image_features, real_image_features)
                        
        return loss


    def cal_l2_loss(self, gen_img, real_img):

        gen_image_features = gen_img['image']
        real_image_features = real_img['image']
        
        mse = torch.nn.MSELoss(reduction='none')
        loss = mse(gen_image_features, real_image_features)
                        
        return loss
    
    def cal_l1_loss(self, gen_img, real_img):

        gen_image_features = gen_img['image']
        real_image_features = real_img['image']
        
        loss = self.l1(gen_image_features, real_image_features)
                        
        return loss

    def save_image_(self, img, fname, drange):
        lo, hi = drange
        img = np.asarray(img.cpu(), dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

        B, C, H, W = img.shape

        img = img.transpose(2,0,3,1)
        img = img.reshape([H, B*W, C])

        assert C in [1, 3]
        if C == 1:
            PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
        if C == 3:
            PIL.Image.fromarray(img, 'RGB').save(fname)
            
    
    def drop_out_pixels(self, img, drop_as='white', p=None):
   
        B,C,H,W = img.shape
        total_pixels = H*W
        drop_ratio = self.drop_pixel_ratio

        if p is None:
            all_idx = torch.arange(0,total_pixels)
            p = torch.ones_like(all_idx).float()
            # # below is to apply doifferent drop out to difference sample in batch
            # p = torch.ones([B,total_pixels]).float()
        

        # print(total_pixels)
        n = int(total_pixels*drop_ratio)
        replace = False
        drop_idx = p.multinomial(num_samples=n, replacement=replace)
        
        if drop_idx.shape[0] == B:
            for _di in drop_idx:
                assert len(torch.unique(_di))==n # no repeatitve drop
        else:
            assert len(torch.unique(drop_idx))==n
        

        assert drop_as in ['white', 'black']
        if drop_as == 'white':
            drop_as_white = img.detach().clone().flatten(-2)
            drop_as_white[...,drop_idx] = 1
            drop_as_white = drop_as_white.reshape(B,C,128,128)
            result =  drop_as_white

        elif drop_as == 'black':
            drop_as_black = img.detach().clone().flatten(-2)
            drop_as_black[...,drop_idx] = 0
            drop_as_black = drop_as_black.reshape(B,C,128,128)
            result = drop_as_black
        
        # # Aux func to check correctness
        # st()
        # self.save_image_(result, 'drop_pixel_result.png', [-1,1])

        return result
    
    def get_condition(self, real_img):
        if self.discriminator_condition_on_real:
            ri = self.drop_out_pixels(real_img['image'].detach().clone())
        elif self.discriminator_condition_on_projection:
            ri = real_img['projection'].detach().clone()
        else:
            print("Not supported type of condition")

        return ri
        

    def accumulate_gradients(self, phase, real_img, real_c, real_proj, gen_z, gen_gt, gen_c, gen_pc, gen_proj, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        ############# FIXME Oct 25: uncomment below to still enable Greg phase ##################
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        ###############################
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw, 'projection': real_proj}

        gen_gt_img = {'image': gen_gt, 'projection': gen_proj}
        # save_fetched_data((gen_gt.detach().clone() + 1)*127.5, gen_c, gen_pc, (gen_proj.detach().clone() + 1)*127.5, 'gen')
        # st()

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:

            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_pc, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)

                if self.use_patchD:
                    if self.patchD_reg==0:
                        loss_Gmain=torch.tensor([0]).to(gen_c)
                    else:
                        if self.discriminator_condition_on_real: # cat with pixel_dropped image
                            gi = gen_img['image']
                            ri = self.drop_out_pixels(gen_gt_img['image'].detach().clone())
                            img = torch.cat([ri, gi], 1)
                        elif self.discriminator_condition_on_projection: 
                            gi = gen_img['image']
                            ri = gen_gt_img['projection'].detach().clone()
                            img = torch.cat([ri, gi], 1)
                        else: 
                            img = gen_img['image']
                        

                        target = True
                        patch_loss = self.run_patchD(img, target)*self.patchD_reg
                        loss_Gmain = patch_loss
                        # print(f"---------loss_patch_G\t\t(x{self.patchD_reg}): {(patch_loss).sum().item()}-------------")
                        training_stats.report('Loss/G/patch_loss_fake',patch_loss)
                        training_stats.report('Loss/G/loss', loss_Gmain)

                
                else:
                    if self.D.is_conditional_D: # cat with pixel_dropped image
                        gen_img['condition']=self.get_condition(gen_gt_img)


                    gen_logits = self.run_D(gen_img, gen_c,  blur_sigma=blur_sigma)
                    training_stats.report('Loss/scores/fake', gen_logits)
                    training_stats.report('Loss/signs/fake', gen_logits.sign())
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                    training_stats.report('Loss/G/loss', loss_Gmain)

                # chamfer loss
                if self.use_chamfer:
                    chamfer_loss = self.chamfer_loss(gen_c, gen_img, gen_pc, neural_rendering_resolution)
                    chamfer_loss *= self.chamfer_reg
                    loss_Gmain += chamfer_loss
                    print(f"---------loss_chamfer\t(x{self.chamfer_reg}): {(chamfer_loss).sum().item()}-------------")
                    training_stats.report('Loss/G/chamfer_loss', chamfer_loss)

                # perceptual loss
                if self.use_perception:
                    perception_loss = self.cal_perception_loss(gen_img=gen_img, real_img=gen_gt_img)
                    perception_loss = torch.mean(perception_loss, 1, True) * self.perception_reg
                    loss_Gmain += perception_loss
                    print(f"---------loss_perception\t(x{self.perception_reg}): {(perception_loss).sum().item()}-------------")
                
                    training_stats.report('Loss/G/perceptual_loss', perception_loss)

                # L1 loss on the whole gen image
                if self.use_l1:

                    l1_loss = self.cal_l1_loss(gen_img=gen_img, real_img=gen_gt_img)
                    # l1_loss = torch.mean(l1_loss.flatten(1), -1, True) * self.l1_reg
                    l1_loss = torch.mean(l1_loss) * self.l1_reg
                    loss_Gmain += l1_loss
                    # print(f"---------loss_l1\t\t(x{self.l1_reg}): {(l1_loss).sum().item()}-------------")
    
                    training_stats.report('Loss/G/l1_loss_whole', l1_loss)

                # L2 loss on the whole gen image
                if self.use_l2:
                    l2_loss = self.cal_l2_loss(gen_img=gen_img, real_img=gen_gt_img)
                    l2_loss = torch.mean(l2_loss.flatten(1), -1, True) * self.l2_reg
                    loss_Gmain += l2_loss
                    print(f"---------loss_l2\t\t(x{self.l2_reg}): {(l2_loss).sum().item()}-------------")
    
                    training_stats.report('Loss/G/l2_loss_whole', l2_loss)
                
                


            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()
        
        

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':

        
            if swapping_prob is not None: # always None
                # c_swapped = torch.roll(gen_c.clone(), 1, 0)
                # c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
                ######## align pc_gen with c_gen #########
                B, N, pc_dim = gen_pc.shape
                st()
                gen_c_and_pc = torch.cat([gen_c, gen_pc.reshape(B,-1)], dim=-1)
                c_and_pc_swapped = torch.roll(gen_c_and_pc.clone(), 1, 0)
                c_and_pc_gen_conditioning = torch.where(torch.rand([], device=gen_c_and_pc.device) < swapping_prob, c_and_pc_swapped, gen_c_and_pc)
                c_gen_conditioning = c_and_pc_gen_conditioning[...,:-pc_dim]
                pc_gen_conditioning = c_and_pc_gen_conditioning[...,-pc_dim:].reshape(B,N,pc_dim)
                assert (pc_gen_conditioning.shape==gen_pc.shape) and (c_gen_conditioning.shape==gen_c.shape)
                st()

            else:
                c_gen_conditioning = torch.zeros_like(gen_c)
                pc_gen_conditioning = gen_pc

            if self.G.z_from_pc:
                ws = self.G.mapping(None, c_gen_conditioning, gen_pc, update_emas=False)
            else:
                ws = self.G.mapping(gen_z, c_gen_conditioning, None, update_emas=False)

            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            # initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            initial_coordinates = torch.rand((gen_z.shape[0], 1000, 3), device=gen_z.device) * 2 - 1 # hard-code to not use ws
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            #### FIXME: the coordinates fed into mixed_sample are both random, why is this?????
            if isinstance(self.G, VolumeGenerator):
                sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, pc=pc_gen_conditioning,\
                     box_warp=self.G.rendering_kwargs['box_warp'], update_emas=False)['sigma']
                # st()
            else:
                sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)
                pc_gen_conditioning = gen_pc

            if self.G.z_from_pc:
                ws = self.G.mapping(None, c_gen_conditioning, gen_pc, update_emas=False)
            else:
                ws = self.G.mapping(gen_z, c_gen_conditioning, None, update_emas=False)

            # initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front
            initial_coordinates = torch.rand((gen_z.shape[0], 2000, 3), device=gen_z.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            if isinstance(self.G, VolumeGenerator):
                sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, pc=pc_gen_conditioning,\
                     box_warp=self.G.rendering_kwargs['box_warp'], update_emas=False)['sigma']
                # st()
            else:
                sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)
                pc_gen_conditioning = gen_pc
            
            if self.G.z_from_pc:
                ws = self.G.mapping(None, c_gen_conditioning, gen_pc, update_emas=False)
            else:
                ws = self.G.mapping(gen_z, c_gen_conditioning, None, update_emas=False)
            
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((gen_z.shape[0], 1000, 3), device=gen_z.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            if isinstance(self.G, VolumeGenerator):
                sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, pc=pc_gen_conditioning,\
                     box_warp=self.G.rendering_kwargs['box_warp'], update_emas=False)['sigma']
                # st()
            else:
                sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)
                pc_gen_conditioning = gen_pc

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((gen_z.shape[0], 2000, 3), device=gen_z.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            if isinstance(self.G, VolumeGenerator):
                sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, pc=pc_gen_conditioning,\
                     box_warp=self.G.rendering_kwargs['box_warp'], update_emas=False)['sigma']
                # st()
            else:
                sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)
                pc_gen_conditioning = gen_pc

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((gen_z.shape[0], 1000, 3), device=gen_z.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            if isinstance(self.G, VolumeGenerator):
                sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, pc=pc_gen_conditioning,\
                     box_warp=self.G.rendering_kwargs['box_warp'], update_emas=False)['sigma']
                # st()
            else:
                sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth'] and self.patchD_reg!=0:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_pc, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                
                if self.use_patchD:
                    
                    if self.discriminator_condition_on_real: # cat with pixel_dropped image
                        gi = gen_img['image']
                        ri = self.drop_out_pixels(gen_gt_img['image'].detach().clone())
                        img = torch.cat([ri, gi], 1)
                        
                    else: 
                        img = gen_img['image']
                    
                    target = False
                    patch_loss_Dgen = self.run_patchD(img, target)*self.patchD_reg
                    loss_Dgen = patch_loss_Dgen
                    training_stats.report('Loss/D/patch_loss_fake', patch_loss_Dgen)
                    # print(f"---------loss_patch Dfake\t\t(x{self.patchD_reg}): {(patch_loss_Dgen).sum().item()}-------------")
                
                else:
                    if self.D.is_conditional_D: # cat with pixel_dropped image
                        gen_img['condition']=self.get_condition(gen_gt_img)

                    gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                    training_stats.report('Loss/scores/fake', gen_logits)
                    training_stats.report('Loss/signs/fake', gen_logits.sign())
                    loss_Dgen = torch.nn.functional.softplus(gen_logits)
                
                
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth'] and self.patchD_reg!=0:
            loss_Dreal = 0

            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                if self.use_patchD:
                    if self.patchD_reg==0:
                        loss_Dreal += torch.tensor([0]).to(gen_c)
                    else:
                        if self.discriminator_condition_on_real: # cat with pixel_dropped image
                            # gi = real_img_tmp_image # this will do reg for image
                            gi = real_img['image'] # this will do reg for image
                            ri = self.drop_out_pixels(real_img['image'].detach().clone())
                            img = torch.cat([ri, gi], 1)
                            
                        else: 
                            img = real_img_tmp_image
                        
                        target = True
                        patch_loss_Dreal, real_logits = self.run_patchD(img, target, return_pred=True)
                        loss_Dreal += patch_loss_Dreal*self.patchD_reg
                        # print(f"---------loss_patch_Dreal\t\t(x{self.patchD_reg}): {(patch_loss_Dreal*self.patchD_reg).sum().item()}-------------")
                        training_stats.report('Loss/D/patch_loss_real', patch_loss_Dreal)
                
                else:
                    if self.D.is_conditional_D: # cat with pixel_dropped image
                        real_img_tmp['condition']=self.get_condition(real_img)

                    real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma, update_emas=True)
                    training_stats.report('Loss/scores/real', real_logits)
                    training_stats.report('Loss/signs/real', real_logits.sign())
                    _loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', _loss_Dreal)
                    loss_Dreal +=  _loss_Dreal 

                if phase in ['Dmain', 'Dboth']:
                    training_stats.report('Loss/D/loss(gen+real)', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth'] and not self.use_patchD:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
