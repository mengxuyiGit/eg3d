import os
# import gdown
# import shutil
# import tempfile
import subprocess
from ipdb import set_trace as st
import glob
import numpy as np
import PIL.Image
import torch
import argparse

def read_image_(file):
    img = np.array(PIL.Image.open(file))
    img = img.transpose(2,0,1)
    img = img.astype(np.float32) / 127.5 - 1

    return img # normalized to -1~1

def one_to_one_metrics(
    calculate_lpips: bool = False,
    calculate_psnr: bool = False,
    gt_dir: str = None,
    pred_dir: str = None,
    ):

    if calculate_lpips:    
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg') # image should be in RGB, noramlized to [-1,1]
        lpips_all= []
    
    if calculate_psnr:
        mse2psnr = lambda x : -10. * np.log(x) / np.log(10.)
        psnr_all= []

    gt_rgb_files   = glob.glob(f"{gt_dir  }/**_rgb.png")
    pred_rgb_files = glob.glob(f"{pred_dir}/**_rgb.png")

    assert len(gt_rgb_files) == len(pred_rgb_files)
    for pair_idx in range (len(gt_rgb_files)):
        gt   = read_image_(os.path.join(gt_dir, f'{pair_idx:06d}_gt_rgb.png'))
        pred = read_image_(os.path.join(pred_dir, f'{pair_idx:06d}_pred_rgb.png'))
       
        if calculate_lpips:           
            lpips = loss_fn_vgg(torch.tensor(pred), torch.tensor(gt)).mean().item()
            lpips_all.append(lpips)
            print(lpips)

        if calculate_psnr:
            # psnr_all.append( mse2psnr(np.mean((gt[~mask]-img[~mask])**2)))
            # TODO: background is also included in psnr
            psnr_all.append( mse2psnr(np.mean((pred - gt)**2)))
            
    st()
    report = "reported_losses:\t\t "
  
    if calculate_lpips:
        report += f"LPIPS: {np.mean(lpips_all)}\t\t"
    else:
        report += "Lpips Not reported\t\t"
    if calculate_psnr:
        report += f"PSNR: {np.mean(psnr_all)}\t\t"
    else:
        report += "PSNR Not reported\t\t"
    print(report)
    
    return report

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dataset_abs_path", type=str, default=None, required=True)
    parser.add_argument("--network_abs_path", type=str, default=None, required=True)
    parser.add_argument("--outdir_rel_path", type=str, default='metrics_out')
    
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    ### only change these parameters ###
        # EXAMPLES:
        # dataset = '/home/xuyi/Repo/eg3d/dataset_preprocessing/abo/debug_proj_train.zip'
        # network='/home/xuyi/Repo/eg3d/eg3d/pretrained_models/Nov6/abo_dataset-abo_3k_128_train_remove_ws_NO_noise_conditionD_projection_L1_000160.pkl'
        # outdir = os.path.join(dir_path, 'gen_metrics_try')
        #  
    dataset = args.dataset_abs_path
    network = args.network_abs_path
    outdir = os.path.join(dir_path, args.outdir_rel_path)
    num_gpus= args.num_gpus
    batch_size= args.batch_size
    ### ---------------------------- ###

    report_FID=True
    report_PSNR=True
    report_lpips=True

    # print("Generating inference results ...") 
    # cmd = f"python {os.path.join(dir_path, 'metric_loop.py')} "
    # cmd += f"--network={network} --validation_data_zip={dataset} "
    # cmd += f"--save_fid_pairs={report_FID} --report_lpips={report_lpips} --report_psnr={report_PSNR} "
    # cmd += f"--num_gpus={num_gpus} --batch_size={batch_size} --outdir={outdir} "
    # # result = subprocess.run([cmd], shell=True)
    # output = subprocess.check_output([cmd], shell=True)
    # output_split = [line for line in output.splitlines()]

    # fid_dir_gt = output_split[-2].decode("utf-8")
    # fid_dir_gen = output_split[-1].decode("utf-8")
    fid_dir_gt = '/xuyi-fast-vol/Repo-fast/eg3d/eg3d/gen_metrics_try/00006-shapenet_plane_200x200_val-gpus1-batch2/gt'
    fid_dir_gen = '/xuyi-fast-vol/Repo-fast/eg3d/eg3d/gen_metrics_try/00006-shapenet_plane_200x200_val-gpus1-batch2/gen'
    print(f"Dir of GT: {fid_dir_gt}\nDIR of PRED: {fid_dir_gen}")

    
    report = one_to_one_metrics(
        calculate_lpips = report_lpips,
        calculate_psnr = report_PSNR,
        gt_dir = fid_dir_gt,
        pred_dir = fid_dir_gen
    )
        
   
    if report_FID:
        print("Calculating FID ...")
        
        cmd = f"python -m pytorch_fid {fid_dir_gt} {fid_dir_gen} --gpu 0"
        output_FID = subprocess.check_output([cmd], shell=True)
        output_FID_split = [line for line in output_FID.splitlines()]
        # print(output_FID_split)
        metric_FID=output_FID_split[-1].decode("utf-8")
        print(metric_FID)