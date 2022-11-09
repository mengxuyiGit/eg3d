# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tool for creating ZIP/PNG based datasets."""

from array import array
import csv
from email.policy import default
import functools
import gzip
import io
import json
import os
import pickle
import re
import sys
import tarfile
from xmlrpc.client import boolean
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import click
import numpy as np
import PIL.Image
from tqdm import tqdm

from ipdb import set_trace as st
import pandas as pd
import point_cloud_utils as pcu

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def parse_tuple(s: str) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

#----------------------------------------------------------------------------

def open_image_folder(source_dir, *, max_images: Optional[int]):
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]
    
    # filter out 'depth' and 'normal'
    # input_images = [ f for f in input_images if ('depth' not in f) and ('normal' not in f) and ('Image' not in f)]
    input_images = [ f for f in input_images if ('render/r_' in f)]

    # -----------only load the selected images in the txt file---------------
    all_data=[]
    for txt_f in os.listdir(os.path.join(source_dir, 'meta')):
        if SPLIT not in txt_f: continue
        with open(os.path.join(source_dir, 'meta', txt_f)) as f:
            scans = [line.rstrip() for line in f.readlines()]
            all_data += scans
    # print(len(all_data), all_data)
    print(len(all_data))
    input_images = [f for f in input_images if f.split(os.sep)[-3] in all_data]
    # st()
    # ----------------------------------------------------------------------------------
 
    # Load labels.
    labels = {}
    meta_fname = os.path.join('/home/xuyi/Repo/eg3d/dataset_preprocessing/abo', 'dataset.json')
    # meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            # st()
            if labels is not None:
                try:
                  
                    # pc_rel_paths = { x[0]: x[2] for x in labels }
                    # pc_rel_paths = { os.path.relpath(x[0], '../output_cutter') : x[2] for x in labels }
                    pc_rel_paths = { os.path.join(x[0].split(os.sep)[-3],x[0].split(os.sep)[-2],x[0].split(os.sep)[-1]): x[2] for x in labels}
         
                    
                    # print(pc_rel_paths)
                except:
                    print("No pointcloud input in dataset")
                    pc_rel_paths = {}
                labels = {os.path.join(x[0].split(os.sep)[-3],x[0].split(os.sep)[-2],x[0].split(os.sep)[-1]) : x[1] for x in labels }
                
            else:
                labels = {}
    # print(labels)
    
    # max_idx = maybe_min(len(input_images), max_images)
    max_idx = maybe_min(len(labels), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            img = np.array(PIL.Image.open(fname))

            if READ_POINTCLOUD:
      
                pc_rel = pc_rel_paths.get(arch_fname[:-4])
               
                if pc_rel != None:
                    pc_fname = os.path.join(source_dir, pc_rel)
                    pc_df = pd.read_csv(pc_fname)
                    pc_array = pc_df[['x','y','z','r','g','b','a', 'metallic','roughness']].values.astype(np.float32)
                    # reda pc csv
                    # save as np array

                    if pc_array.shape[0] > NUM_POINTS: # poisson sampling
                        st()
                        n_sample_poisson = NUM_POINTS
                        particle_pos = pc_array[:, :3]
                        poisson_idx = pcu.downsample_point_cloud_poisson_disk(particle_pos, num_samples=n_sample_poisson)
                        while poisson_idx.shape[0] < NUM_POINTS:
                            n_sample_poisson += 50
                            poisson_idx = pcu.downsample_point_cloud_poisson_disk(particle_pos, num_samples=n_sample_poisson)
                        poisson_idx = poisson_idx[:NUM_POINTS]
                        # particle_pos = particle_pos[poisson_idx]
                        pc_array = pc_array[poisson_idx]    
                else:
                    print("continue")
                    print(arch_fname[:-4])
                    continue
            else:
                pc_array=None
            
            if READ_PROJECTION:
      
                # proj_rel = pc_rel_paths.get(arch_fname[:-4])
                # proj_rel = arch_fname.replace('r_', '128_1024/pc_1024_')
                # st()
                proj_rel = arch_fname.replace('r_', f'{RESOLUTION}_1024/pc_1024_')
               
                if proj_rel != None:
                    proj_fname = os.path.join(source_dir, proj_rel)
                    proj_img = np.array(PIL.Image.open(proj_fname))
                
                else:
                    print("projection not exist, continue")
                    print(proj_fname[:-4])
                    continue
            else:
                proj_img=None

            
            arch_fname = os.path.splitext(arch_fname)[0]
            label_get = labels.get(arch_fname)
       
            if label_get != None:
                # st()
                # print('fname, labels.get(fname)', arch_fname)
                yield dict(img=img, label=labels.get(arch_fname), pc=pc_array, proj_img=proj_img)
            else:
                # print(arch_fname, ' is not written into json file yet') # in the abo case
                # st() 
                pass
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]
        st()
        # Load labels.
        labels = {}
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                labels = json.load(file)['labels']
                if labels is not None:
                    labels = { x[0]: x[1] for x in labels }
                else:
                    labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = PIL.Image.open(file) # type: ignore
                    img = np.array(img)
            
                yield dict(img=img, label=labels.get(fname))
                if idx >= max_idx-1:
                    break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_lmdb(lmdb_dir: str, *, max_images: Optional[int]):
    import cv2  # pip install opencv-python # pylint: disable=import-error
    import lmdb  # pip install lmdb # pylint: disable=import-error

    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        max_idx = maybe_min(txn.stat()['entries'], max_images)

    def iterate_images():
        with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.array(PIL.Image.open(io.BytesIO(value)))
                    yield dict(img=img, label=None)
                    if idx >= max_idx-1:
                        break
                except:
                    print(sys.exc_info()[1])

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_cifar10(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:
        for batch in range(1, 6):
            member = tar.getmember(f'cifar-10-batches-py/data_batch_{batch}')
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')
            images.append(data['data'].reshape(-1, 3, 32, 32))
            labels.append(data['labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1]) # NCHW -> NHWC
    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_mnist(images_gz: str, *, max_images: Optional[int]):
    labels_gz = images_gz.replace('-images-idx3-ubyte.gz', '-labels-idx1-ubyte.gz')
    assert labels_gz != images_gz
    images = []
    labels = []

    with gzip.open(images_gz, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    with gzip.open(labels_gz, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
 
        if width == w and height == h:
            return img
        st()
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int]):
    # st()
    if os.path.isdir(source):
        if source.rstrip('/').endswith('_lmdb'):
            return open_lmdb(source, max_images=max_images)
        else:
            return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if os.path.basename(source) == 'cifar-10-python.tar.gz':
            return open_cifar10(source, max_images=max_images)
        elif os.path.basename(source) == 'train-images-idx3-ubyte.gz':
            return open_mnist(source, max_images=max_images)
        elif file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            assert False, 'unknown archive type'
    else:
        error(f'Missing input file or directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)

        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)
        st()

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--max-images', help='Output only up to `max-images` images', type=int, default=None)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--resolution', help='Output resolution (e.g., \'512x512\')', metavar='WxH', type=parse_tuple)
@click.option('--read_pointcloud', help='whether pc.csv is in dataset)', type=boolean, is_flag=True, default=False)
@click.option('--split', help='Directory or archive name for input dataset', required=True, metavar='PATH')

def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]],
    read_pointcloud:Optional[boolean],
    split: str,
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                    Load LSUN dataset
    --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
    --source train-images-idx3-ubyte.gz Load MNIST dataset
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompressed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the dataset
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, the dataset is interpreted as
    not containing class labels.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --resolution option.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --resolution=512x384
    """

    if read_pointcloud:
        global READ_POINTCLOUD
        READ_POINTCLOUD = True
        global NUM_POINTS
        NUM_POINTS = 1024
    
    global READ_PROJECTION
    READ_PROJECTION = True
    
    global SPLIT
    SPLIT = split

    PIL.Image.init() # type: ignore

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    
    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    if resolution is None: resolution = (None, None)
    transform_image = make_transform(transform, *resolution)
    global RESOLUTION
    RESOLUTION = resolution[0]

    dataset_attrs = None

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        
        
        # print(image)
        
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        # Apply crop and resize.
        img = transform_image(image['img'])
        
        # Transform may drop images.
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': channels
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if dataset_attrs['channels'] not in [1, 3, 4]:
                error('Input images must be stored as RGB or grayscale')
        
            # if width != 2 ** int(np.floor(np.log2(width))):
            #     error('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()] # pylint: disable=unsubscriptable-object
            error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        WHITE_BKGD=True
        if channels == 4 and WHITE_BKGD:
            img = img[...,:3] * (img[...,-1:]/255) + (255 - img[...,-1:])
            img = img.astype(np.uint8)
            channels = 3
            # im1 = img.save("geeks_white.png")
            # st()
        img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB', 4: 'RGBA'}[channels])
        # img.save("dataset_tool_img.png")
        
        if not WHITE_BKGD:
            if channels == 4: img = img.convert('RGB')
            # im1 = img.save("geeks_blk.png")
            # st()

        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        # print(archive_fname)
        labels.append([archive_fname, image['label']] if image['label'] is not None else None)

        # Save pc.csv also using the same indexed archname
        pc_array = image['pc']
        archive_fname_pc = archive_fname.replace('img', 'pc').replace('png', 'csv')
        string_buffer = io.StringIO()
        writer = csv.writer(string_buffer)
        for row in pc_array:
            writer.writerow(row)
        save_bytes(os.path.join(archive_root_dir, archive_fname_pc), string_buffer.getvalue())
        # print(archive_fname_pc)

        # save projection as condition
        proj_img = transform_image(image['proj_img'])
        channels = proj_img.shape[2] if proj_img.ndim == 3 else 1
        # PIL.Image.fromarray(proj_img, { 1: 'L', 3: 'RGB', 4: 'RGBA'}[channels]).save("dataset_tool_proj.png")
        if channels == 4 and WHITE_BKGD:
            proj_img = proj_img[...,:3] * (proj_img[...,-1:]/255) + (255 - proj_img[...,-1:])
            proj_img = proj_img.astype(np.uint8)
            channels = 3
        
        if not WHITE_BKGD:
            if channels == 4: proj_img = proj_img.convert('RGB')
        
        proj_img = PIL.Image.fromarray(proj_img, { 1: 'L', 3: 'RGB', 4: 'RGBA'}[channels])
        # proj_img.save("dataset_tool_proj.png")
        # st()
        

        proj_image_bits = io.BytesIO()
        proj_img.save(proj_image_bits, format='png', compress_level=0, optimize=False)
        archive_fname_proj = archive_fname.replace('img', 'proj_img')
        save_bytes(os.path.join(archive_root_dir, archive_fname_proj), proj_image_bits.getbuffer())
        # print(archive_fname, archive_fname_pc, archive_fname_proj)
       

    metadata = {
        'labels': labels if all(x is not None for x in labels) else None
    }
    # print(metadata)
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    # print(os.path.join(archive_root_dir, 'dataset.json'))
    close_dest()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter
