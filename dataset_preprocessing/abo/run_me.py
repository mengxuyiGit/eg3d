# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import gdown
import shutil
import tempfile
import subprocess


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as working_dir:
        # working_dir = '/tmp/tmphal02_sj' # /cars_train.zip
        working_dir = '/home/xuyi/Data'
        # print(working_dir)
        # download_name = 'cars_train.zip'
        # url = 'https://drive.google.com/uc?id=1bThUNtIHx4xEQyffVBSf82ABDDh2HlFn'
        # output_dataset_name = 'abo_128_completed.zip'
        # output_dataset_name = 'abo_128_completed_white.zip'
        # output_dataset_name = 'abo_512_completed_white.zip'
        # output_dataset_name = 'debug.zip'
        npoints = 2048
        res = 128
        target_dataset_npoints_res = f'abo_{npoints}_{res}'
        output_dataset_name = f'{target_dataset_npoints_res}_completed_white.zip'

        dir_path = os.path.dirname(os.path.realpath(__file__))
        
        extracted_data_path = '/home/xuyi/Data/renderer/output_cutter'

        print("Converting camera parameters...")
        cmd = f"python {os.path.join(dir_path, 'preprocess_abo_cameras.py')} --source={extracted_data_path} --file_prefix={target_dataset_npoints_res}"
        subprocess.run([cmd], shell=True)

        print("Creating dataset zip...")
        cmd = f"python {os.path.join(dir_path, '../../eg3d', 'dataset_tool.py')}"
        cmd += f" --source {extracted_data_path} --dest {output_dataset_name} --resolution {res}x{res} --read_pointcloud --num_points {npoints}"
        subprocess.run([cmd], shell=True)