import os
import gdown
import shutil
import tempfile
import subprocess
from ipdb import set_trace as st


if __name__ == '__main__':
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    dataset = '/home/xuyi/Repo/eg3d/dataset_preprocessing/abo/debug_proj_train.zip'
    # '/home/xuyi/Repo/eg3d/try-runs/00690-abo_dataset-debug_one_obj_one_view-gpus1-batch1-gamma0.3/network-snapshot-000000.pkl'
    network='/home/xuyi/Repo/eg3d/eg3d/pretrained_models/Nov6/abo_dataset-abo_3k_128_train_remove_ws_NO_noise_conditionD_projection_L1_000160.pkl'
    outdir = os.path.join(dir_path, 'gen_metrics_try')

    num_gpus=1
    batch_size=2
    report_FID=True
    report_L1=True
    report_PSNR=True
    report_lpips=True

    print("Generating inference results ...") 
    cmd = f"python {os.path.join(dir_path, 'gen_loop.py')} "
    cmd += f"--network={network} --validation_data_zip={dataset} "
    cmd += f"--save_fid_pairs={report_FID} --report_l1_loss={report_L1} --report_lpips={report_lpips} --report_psnr={report_PSNR} "
    cmd += f"--num_gpus={num_gpus} --batch_size={batch_size} --outdir={outdir} "
    # result = subprocess.run([cmd], shell=True)
    output = subprocess.check_output([cmd], shell=True)
    output_split = [line for line in output.splitlines()]

    if report_L1 and report_FID:
        metric_L1 = output_split[-4].decode("utf-8")
        print(metric_L1)
    if report_FID:
        print("Calculating FID ...")
        fid_dir_gt = output_split[-2].decode("utf-8")
        fid_dir_gen = output_split[-1].decode("utf-8")
        print(f"FID gt: {fid_dir_gt} gen: {fid_dir_gen}")
        cmd = f"python -m pytorch_fid {fid_dir_gt} {fid_dir_gen} --gpu 0"
        output_FID = subprocess.check_output([cmd], shell=True)
        output_FID_split = [line for line in output_FID.splitlines()]
        # print(output_FID_split)
        metric_FID=output_FID_split[-1].decode("utf-8")
        print(metric_FID)