# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import dill as pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main
from camera_utils import LookAtPoseSampler
from training.crosssection_utils import sample_cross_section

from ipdb import set_trace as st
from plyfile import PlyData,PlyElement
from typing import List, Optional, Tuple, Union
from training.volume import VolumeGenerator
import argparse
from train import init_dataset_kwargs
import re
from tqdm import tqdm


#-------GEN_VIDEO---------------------------------------------------------------------

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

# NO THE SAME AS LOADING MODEL FOR TRAINING
def copy_params_and_buffers(src_module, dst_module, device, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
            tensor.to(device)
#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

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
        label_order = list(label_groups.keys())
        rnd.shuffle(label_order)
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.

    images, labels, pc_arrays, projection = zip(*[training_set[i] for i in grid_indices])

    # print('getitem types ---------------------->',type(images[0]), type(labels[0]), type(pc_arrays[0]))
    return (gw, gh), np.stack(images), np.stack(labels), np.stack(pc_arrays)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def save_ply_from_tensor(points, save_path, color=None, write_text=False):
    if color != None:
        pts_color = torch.tensor(color, device=points.device).repeat(points.shape[0],1)
        points = torch.cat((points, pts_color), dim=-1)

    assert points.shape[-1]>=6
    pts = points.numpy()
    n = pts.shape[0]
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    pts = (pts*255).astype(int)
    red, green, blue = pts[:,3], pts[:,4], pts[:,5]

    # connect the proper data structures
    vertices = np.empty(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices['x'] = x.astype('f4')
    vertices['y'] = y.astype('f4')
    vertices['z'] = z.astype('f4')
    vertices['red'] = red.astype('u1')
    vertices['green'] = green.astype('u1')
    vertices['blue'] = blue.astype('u1')

    # save as ply
    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
    ply.write(save_path)
    # print("Ply file saved to:", save_path)
    return 



def save_fetched_data(phase_real_img, phase_real_c, phase_real_pc, data_idx):
    # print(phase_real_img, phase_real_c, phase_real_pc)
    
    if phase_real_img.dim()>3:
        for i, (img, c, pc) in enumerate(zip(phase_real_img, phase_real_c, phase_real_pc)):
        # for img, c, pc in zip(phase_real_img, phase_real_c, phase_real_pc):
            save_path = os.path.join("/home/xuyi/Repo/eg3d/try-runs/debug_data", f"data_{data_idx}_{i}")
            print(save_path)
            PIL.Image.fromarray(img.detach().clone().cpu().numpy().astype(np.uint8).transpose(1,2,0), 'RGB').save(f"{save_path}.png")
            save_ply_from_tensor(pc.detach().clone().cpu(), f"{save_path}.ply")
            print(c)
       
    # st()
    return 

def save_image_(img, fname, drange):
    lo, hi = drange
    img = np.asarray(img.cpu(), dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    
    C, H, W = img.shape

    img = img.transpose(1,2,0)
    img = img.reshape([H, W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

    return 
#----------------------------------------------------------------------------

def inference_loop(
    ### below is from gen_video 
    calculate_l1: bool = False,
    calculate_lpips: bool = False,
    calculate_psnr: bool = False,
    save_fid_pairs: bool = False,
    sampling_multiplier: float = 2,
    nrr: Optional[int] = None,

    #### below is from trianing loop
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    network_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.

    
):
    
    os.makedirs(run_dir, exist_ok=True)
    print("Run dir:", run_dir)

    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # # Construct networks.
    # if rank == 0:
    #     print('Constructing networks...')
    # common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    # G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    # # G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G).to(device)
    # G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    
    # D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    # # D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D).to(device)
    # G_ema = copy.deepcopy(G).eval()
    # # print('---------------------> G_kwargs', G_kwargs)
    # # print('---------------------> common_kwargs', common_kwargs)

    # # Resume from existing pickle.
    # if (resume_pkl is not None) and (rank == 0):
    #     print(f'Resuming from "{resume_pkl}"')
    #     with dnnlib.util.open_url(resume_pkl) as f:
    #         resume_data = legacy.load_network_pkl(f)
    #     for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
    #         # module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module).to(device)
    #         misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    if rank == 0:
    #     print('Constructing networks...')
        print('Loading networks from "%s"...' % network_pkl)
    # device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)# type: ignore
        init_kwargs = G.init_kwargs
        t = VolumeGenerator(**init_kwargs).eval().to(device)
        with torch.no_grad():
            copy_params_and_buffers(G, t, device, require_all=True)
        G = t


    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr

    # if truncation_cutoff == 0:
    #     truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    # if truncation_psi == 1.0:
    #     truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff
    # ------------------LOADED MODEL TO INFERENCE------------------------

    # # Print network summary tables.
    # if rank == 0:
    #     z = torch.empty([batch_gpu, G.z_dim], device=device)
    #     c = torch.empty([batch_gpu, G.c_dim], device=device)

    #     from training.volume import VolumeGenerator
    #     if isinstance(G, VolumeGenerator):
    #         pc = torch.empty([batch_gpu]+ [i for i in G.pc_dim], device=device) # (4, 1500, 9)
    #         img = misc.print_module_summary(G, [z, c, pc])
    #     else:
    #         img = misc.print_module_summary(G, [z, c])

    #     from training.patch_discriminator import PatchDiscriminator
    #     if isinstance(D, PatchDiscriminator):
    #         img_d = img['image']
    #         if loss_kwargs.discriminator_condition_on_real:
    #             img_d = torch.cat([img_d, img_d], 1) # [B,6,H,W]
    #         misc.print_module_summary(D, [img_d, True])
    #     else:
    #         misc.print_module_summary(D, [img, c])

    # # Setup augmentation.
    # if rank == 0:
    #     print('Setting up augmentation...')
    # augment_pipe = None
    # ada_stats = None
    # if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
    #     augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    #     augment_pipe.p.copy_(torch.as_tensor(augment_p))
    #     if ada_target is not None:
    #         ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    # for module in [G, D, G_ema, augment_pipe]:
    for module in [G]:
        if module is not None:
            # module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module).to(device)
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # # Setup training phases.
    # if rank == 0:
    #     print('Setting up training phases...')
    # loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    # phases = []
    # for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
    #     if reg_interval is None:
    #         # print([name for name, p in module.named_parameters() if p.requires_grad]) # : empty
    #         # opt = dnnlib.util.construct_class_by_name(params=[p for p in module.parameters() if p.requires_grad], **opt_kwargs) # subclass of torch.optim.Optimizer
    #         opt = dnnlib.util.construct_class_by_name(params= module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
    #         phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
    #     else: # Lazy regularization.
    #         mb_ratio = reg_interval / (reg_interval + 1)
    #         opt_kwargs = dnnlib.EasyDict(opt_kwargs)
    #         opt_kwargs.lr = opt_kwargs.lr * mb_ratio
    #         opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
    #         # print([name for name, p in module.named_parameters() if p.requires_grad]) # : empty
    #         # opt = dnnlib.util.construct_class_by_name(params=[p for p in module.parameters() if p.requires_grad], **opt_kwargs) # subclass of torch.optim.Optimizer
    #         # TODO: check D contains patchD
    #         opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
    #         phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
    #         phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    # for phase in phases:
    #     phase.start_event = None
    #     phase.end_event = None
    #     if rank == 0:
    #         phase.start_event = torch.cuda.Event(enable_timing=True)
    #         phase.end_event = torch.cuda.Event(enable_timing=True)
    # print("Training phases:\n", [p.name for p in phases])

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    grid_pc = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels, pointclouds = setup_snapshot_image_grid(training_set=training_set)
        print("------------->", grid_size, images.shape, labels.shape, pointclouds.shape)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        grid_pc = torch.from_numpy(pointclouds).to(device).split(batch_gpu)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # # Train.
    # if rank == 0:
    #     print(f'Training for {total_kimg} kimg...')
    #     print()
    # cur_nimg = resume_kimg * 1000
    # cur_tick = 0
    # tick_start_nimg = cur_nimg
    # tick_start_time = time.time()
    # maintenance_time = tick_start_time - start_time
    # batch_idx = 0
    # if progress_fn is not None:
    #     progress_fn(0, total_kimg)
    
    DEBUG_DATA=True
    if DEBUG_DATA:
        global data_idx
        data_idx = 0

    if calculate_l1:
        # get_l1 = torch.nn.L1Loss()
        l1_losses = []
    
    if calculate_lpips:    
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg') # image should be in RGB, noramlized to [-1,1]
        lpips_all= []
    
    if calculate_psnr:
        mse2psnr = lambda x : -10. * np.log(x) / np.log(10.)
        psnr_all= []

    if save_fid_pairs:
        fid_dir_gt = os.path.join(run_dir, 'gt')
        os.makedirs(fid_dir_gt, exist_ok=True)
        fid_dir_gen = os.path.join(run_dir, 'gen')
        os.makedirs(fid_dir_gen, exist_ok=True)
        fid_dir_cat = os.path.join(run_dir, 'gen_gt_cat')
        os.makedirs(fid_dir_cat, exist_ok=True)
        fid_pair_idx = 0

        
    while True: # only once! no looop
        with torch.no_grad(): # inference, no grad faster!

        # Fetch training data.
        # with torch.autograd.profiler.record_function('data_fetch'):
            # # phase_real_img, phase_real_c, phase_real_pc = next(training_set_iterator)
            # # if DEBUG_DATA:
            # #     save_fetched_data(phase_real_img, phase_real_c, phase_real_pc, data_idx)
            # #     data_idx += 1
            # #     # continue

            # phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            # phase_real_c = phase_real_c.to(device).split(batch_gpu)
            # phase_real_pc = phase_real_pc.to(device).split(batch_gpu)

            
            # gen_indices = [np.random.randint(len(training_set)) for _ in range(len(phases) * batch_size)]
            gen_indices = [i for i in range(len(training_set))]
            
            # all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = torch.randn([len(gen_indices), G.z_dim], device=device)

            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            # same indices for c and pc
            # ADD: load all_gen_gt_img
            all_gen_gt = [training_set.get_image(idx) for idx in gen_indices]
            all_gen_gt = torch.from_numpy(np.stack(all_gen_gt)).pin_memory().to(device)
            all_gen_gt = (all_gen_gt.to(torch.float32) / 127.5 - 1) # 0~255 -> -1~1
            all_gen_gt = [phase_gen_gt.split(batch_gpu) for phase_gen_gt in all_gen_gt.split(batch_size)]

            all_gen_c = [training_set.get_label(idx) for idx in gen_indices]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]
            all_gen_pc = [training_set.get_pointcloud(idx) for idx in gen_indices]
            all_gen_pc = torch.from_numpy(np.stack(all_gen_pc)).pin_memory().to(device)
            all_gen_pc = [phase_gen_pc.split(batch_gpu) for phase_gen_pc in all_gen_pc.split(batch_size)]
        
     

        # Execute training phases.
        # for phase, phase_gen_z, phase_gen_gt, phase_gen_c, phase_gen_pc in zip(phases, all_gen_z, all_gen_gt, all_gen_c, all_gen_pc):
        for phase_gen_z, phase_gen_gt, phase_gen_c, phase_gen_pc in tqdm(zip(all_gen_z, all_gen_gt, all_gen_c, all_gen_pc), total=len(gen_indices)//batch_gpu):
            # if batch_idx % phase.interval != 0:
            #     continue
            # if phase.start_event is not None:
            #     phase.start_event.record(torch.cuda.current_stream(device))

            # # Accumulate gradients.
            # phase.opt.zero_grad(set_to_none=True)
            # phase.module.requires_grad_(True)

            # if rank == 0:
            #     print('############# current phase:', phase, '############')

            # if DEBUG_DATA:
                
            #     save_fetched_data((phase_real_img[0].detach().clone() + 1)*127.5, phase_real_c[0], phase_real_pc[0], data_idx)
            #     data_idx += 1
            #     # continue
            #     st()

            # for real_img, real_c, gen_z, gen_gt, gen_c, gen_pc in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_gt, phase_gen_c, phase_gen_pc):
            #     loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_gt=gen_gt, gen_c=gen_c, gen_pc=gen_pc, gain=phase.interval, cur_nimg=cur_nimg)
            # if 'G' in phase:
            #     st() # check patchD
            # phase.module.requires_grad_(False)
            neural_rendering_resolution=128 

            for gen_z, gen_gt, gen_c, gen_pc in zip(phase_gen_z, phase_gen_gt, phase_gen_c, phase_gen_pc):
                # loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_gt=gen_gt, gen_c=gen_c, gen_pc=gen_pc, gain=phase.interval, cur_nimg=cur_nimg)
                # out = [G(z=z, c=c, pc=pc, noise_mode='const') for z, c, pc in zip(grid_z, grid_c, grid_pc)]
                ws = G.mapping(gen_z, gen_c, None)
                # print("ws after G mapping:", ws) # None, passed
    
                gen_output = G.synthesis(ws, gen_c, gen_pc, neural_rendering_resolution=neural_rendering_resolution)
                # print("gen img norm range", gen_output['image'].min(), gen_output['image'].max(), gen_gt.min(), gen_gt.max()) :confirmed: -1~1
              
                pred, gt = torch.clamp(gen_output['image'].cpu(),-1,1.0).numpy(), gen_gt.cpu().numpy()
                
                if calculate_l1:
                    l1_loss = np.mean((np.abs(pred - gt)))
                    l1_losses.append(l1_loss)
                    training_stats.report('Loss/G/l1_loss_whole', l1_loss)

                if calculate_lpips:
                    
                    lpips = loss_fn_vgg(torch.tensor(pred), torch.tensor(gt)).mean().item()
                    # print("lpips",lpips)
                    # exit()
                    lpips_all.append(lpips)
                    pass

                if calculate_psnr:
                    # psnr_all.append( mse2psnr(np.mean((gt[~mask]-img[~mask])**2)))
                    # FIXME: background is also included in psnr
                    psnr_all.append( mse2psnr(np.mean((pred - gt)**2)))
                    pass

                if save_fid_pairs:
                    gen_gt_pairs=[]
                    for fid_gen, fid_gt in zip(gen_output['image'], gen_gt):
                        save_image_(fid_gen, os.path.join(fid_dir_gen, f'{fid_pair_idx:06d}_gen_rgb.png'), drange=[-1,1])
                        save_image_(fid_gt, os.path.join(fid_dir_gt, f'{fid_pair_idx:06d}_gt_rgb.png'), drange=[-1,1])
                        if fid_pair_idx%1==0:
                            gen_gt_pairs.append(torch.cat([fid_gt, fid_gen], dim=-1))
                            # save_image_(torch.cat([fid_gt, fid_gen], dim=-1), os.path.join(fid_dir_cat, f'{fid_pair_idx:06d}_gt_rgb.png'), drange=[-1,1])
                        fid_pair_idx += 1
                        # print(f"FID pair {fid_pair_idx} saved!")

                    gen_gt_pairs = torch.stack(gen_gt_pairs, dim=0).unqueeze(0)
                    N, C, H ,W = gen_gt_pairs.shape
                    print(.reshape().shape)
                    exit()
            
            # if 'G' in phase:
            #     st() # check patchD
            # phase.module.requires_grad_(False)

            # # Update weights.
            # with torch.autograd.profiler.record_function(phase.name + '_opt'):
            #     params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
            #     if len(params) > 0:
            #         flat = torch.cat([param.grad.flatten() for param in params])
            #         if num_gpus > 1:
            #             torch.distributed.all_reduce(flat)
            #             flat /= num_gpus
            #         misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            #         grads = flat.split([param.numel() for param in params])
            #         for param, grad in zip(params, grads):
            #             param.grad = grad.reshape(param.shape)
            #     phase.opt.step()

            # # Phase done.
            # if phase.end_event is not None:
            #     phase.end_event.record(torch.cuda.current_stream(device))

        # # Update G_ema.
        # with torch.autograd.profiler.record_function('Gema'):
        #     ema_nimg = ema_kimg * 1000
        #     if ema_rampup is not None:
        #         ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
        #     ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
        #     for p_ema, p in zip(G_ema.parameters(), G.parameters()):
        #         p_ema.copy_(p.lerp(p_ema, ema_beta))
        #     for b_ema, b in zip(G_ema.buffers(), G.buffers()):
        #         b_ema.copy_(b)
        #     G_ema.neural_rendering_resolution = G.neural_rendering_resolution
        #     G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # # Update state.
        # cur_nimg += batch_size
        # batch_idx += 1

        # # Execute ADA heuristic.
        # if (ada_stats is not None) and (batch_idx % ada_interval == 0):
        #     ada_stats.update()
        #     adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
        #     augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # # Perform maintenance tasks once per tick.
        # done = (cur_nimg >= total_kimg * 1000)
        # if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
        # # if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 2):
        #     continue

        # # Print status line, accumulating the same information in training_stats.
        # tick_end_time = time.time()
        # fields = []
        # fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        # fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        # fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        # fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        # fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        # fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        # fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        # fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        # fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        # torch.cuda.reset_peak_memory_stats()
        # fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        # training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        # training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        # if rank == 0:
        #     print(' '.join(fields))

        # # Check for abort.
        # if (not done) and (abort_fn is not None) and abort_fn():
        #     done = True
        #     if rank == 0:
        #         print()
        #         print('Aborting...')

        # # Save image snapshot.
        # if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
        #     if all(x != 0 for x in pointclouds.shape):
        #         # st()
        #         out = [G_ema(z=z, c=c, pc=pc, noise_mode='const') for z, c, pc in zip(grid_z, grid_c, grid_pc)]
        #     else:
        #         st()
        #         out = [G_ema(z=z, c=c, noise_mode='const') for z, c in zip(grid_z, grid_c)]
        #     images = torch.cat([o['image'].cpu() for o in out]).numpy()
        #     images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
        #     images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
        #     save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
        #     save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw.png'), drange=[-1,1], grid_size=grid_size)
        #     save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

            #--------------------
            # # Log forward-conditioned images

            # forward_cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
            # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
            # forward_label = torch.cat([forward_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            # grid_ws = [G_ema.mapping(z, forward_label.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
            # out = [G_ema.synthesis(ws, c=c, noise_mode='const') for ws, c in zip(grid_ws, grid_c)]

            # images = torch.cat([o['image'].cpu() for o in out]).numpy()
            # images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            # images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            # save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_f.png'), drange=[-1,1], grid_size=grid_size)
            # save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw_f.png'), drange=[-1,1], grid_size=grid_size)
            # save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth_f.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

            #--------------------
            # # Log Cross sections

            # grid_ws = [G_ema.mapping(z, c.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
            # out = [sample_cross_section(G_ema, ws, w=G.rendering_kwargs['box_warp']) for ws, c in zip(grid_ws, grid_c)]
            # crossections = torch.cat([o.cpu() for o in out]).numpy()
            # save_image_grid(crossections, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_crossection.png'), drange=[-50,100], grid_size=grid_size)

        # # Save network snapshot.
        # snapshot_pkl = None
        # snapshot_data = None
        # if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
        #     snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
        #     for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
        #         if module is not None:
        #             if num_gpus > 1:
        #                 misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema|mean)')
        #             module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
        #         snapshot_data[name] = module
        #         del module # conserve memory
        #     snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
        #     if rank == 0:
        #         with open(snapshot_pkl, 'wb') as f:
        #             pickle.dump(snapshot_data, f)

        # # Evaluate metrics.
        # if (snapshot_data is not None) and (len(metrics) > 0):
        # # if (snapshot_data is not None) and (len(metrics) > 0) and batch_idx % 10 ==0:
        #     if rank == 0:
        #         print(run_dir)
        #         print('Evaluating metrics...')
        #     for metric in metrics:
        #         result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
        #             dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
        #         if rank == 0:
        #             metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
        #         stats_metrics.update(result_dict.results)
        # del snapshot_data # conserve memory

        # # Collect statistics.
        # for phase in phases:
        #     value = []
        #     if (phase.start_event is not None) and (phase.end_event is not None):
        #         phase.end_event.synchronize()
        #         value = phase.start_event.elapsed_time(phase.end_event)
        #     training_stats.report0('Timing/' + phase.name, value)
        # stats_collector.update()
        # stats_dict = stats_collector.as_dict()

        # # Update logs.
        # timestamp = time.time()
        # if stats_jsonl is not None:
        #     fields = dict(stats_dict, timestamp=timestamp)
        #     stats_jsonl.write(json.dumps(fields) + '\n')
        #     stats_jsonl.flush()
        # if stats_tfevents is not None:
        #     global_step = int(cur_nimg / 1e3)
        #     walltime = timestamp - start_time
        #     for name, value in stats_dict.items():
        #         stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
        #     for name, value in stats_metrics.items():
        #         stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
        #     stats_tfevents.flush()
        # if progress_fn is not None:
        #     progress_fn(cur_nimg // 1000, total_kimg)

        # # Update state.
        # cur_tick += 1
        # tick_start_nimg = cur_nimg
        # tick_start_time = time.time()
        # maintenance_time = tick_start_time - tick_end_time
        # # if done:
        break # break when looped for once 

    # Done.
    if rank == 0:
        print()
        report = "reported_losses:\t\t "
        if calculate_l1:
            # l1_losses_mean = f"L1: {np.mean(l1_losses)}\t\t"
            report += f"L1: {np.mean(l1_losses)}\t\t"
        else:
            report += "L1 Not reported\t\t"   
        if calculate_lpips:
            report += f"LPIPS: {np.mean(lpips_all)}\t\t"
        else:
            report += "Lpips Not reported\t\t"
        if calculate_psnr:
            report += f"PSNR: {np.mean(psnr_all)}\t\t"
        else:
            report += "PSNR Not reported\t\t"
        print(report)

        if save_fid_pairs:
            print(f"FID pair {fid_pair_idx} saved!")
            print(fid_dir_gt)
            print(fid_dir_gen)

        # print('Exiting...')
        
    return run_dir

#----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--source", type=str)
    # parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--save_fid_pairs", type=bool, default=False)
    parser.add_argument("--report_l1_loss", type=bool, default=False)
    parser.add_argument("--report_lpips", type=bool, default=False)
    parser.add_argument("--report_psnr", type=bool, default=False)
    parser.add_argument("--validation_data_zip", type=str, default=None)
    parser.add_argument("--network", type=str, default=None, required=True)
    
    args = parser.parse_args()

    training_set_kwargs, dataset_name = init_dataset_kwargs(data=args.validation_data_zip)
    print(f'Dataset path:        {training_set_kwargs.path}')
    print(f'Dataset size:        {training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {training_set_kwargs.resolution}')
    print(f'Dataset labels:      {training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {training_set_kwargs.xflip}')

    # Pick output directory.
    prev_run_dirs = []
    outdir = args.outdir
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    desc = f'{dataset_name:s}-gpus{args.num_gpus:d}-batch{args.batch_size:d}'
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(run_dir)
    

    with torch.no_grad():
        inference_loop(
            ### below is from gen_video
            calculate_l1 = args.report_l1_loss,
            calculate_lpips = args.report_lpips,
            calculate_psnr = args.report_psnr,
            save_fid_pairs=args.save_fid_pairs,
            training_set_kwargs=training_set_kwargs,
            run_dir=run_dir,
            num_gpus=args.num_gpus,        # Number of GPUs participating in the training.
            batch_size = args.batch_size,
            batch_gpu = args.batch_size // args.num_gpus, 
            network_pkl=args.network,
            )
        # return run_dir
        