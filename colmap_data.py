#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path, PurePosixPath
import transform_colmap_camera

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="convert transforms.json to colmap.txt")

    parser.add_argument("--width", default="4036", help="image width")
    parser.add_argument("--height", default="3024", help="image height")
    parser.add_argument("--images", default="images", help="input path to the images")
    parser.add_argument("--colmap_db", default="database.db", help="colmap database filename")
    args = parser.parse_args()
    return args

def do_system(arg):
    print(f"==== running: {arg}")
    err=os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def go_camera2colmap(args):
    # TODO: change image size
    H = int(args.height)
    W = int(args.width)
    
    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # 注意：最后输出的图片名字要按自然字典序排列，例：0, 1, 100, 101, 102, 2, 3...因为colmap内部是这么排序的
    fnames = list(sorted(os.listdir('images')))
    fname2pose = {}
    
    with open('transforms.json', 'r') as f:
        meta = json.load(f)
    
    k1, k2, p1, p2 = meta['k1'], meta['k2'], meta['p1'], meta['p2']
    fx = 0.5 * W / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
    if 'camera_angle_y' in meta:
        fy = 0.5 * H / np.tan(0.5 * meta['camera_angle_y'])  # original focal length
    else:
        fy = fx
    if 'cx' in meta:
        cx, cy = meta['cx'], meta['cy']
    else:
        cx = 0.5 * W
        cy = 0.5 * H
    with open('sparse/0/cameras.txt', 'w') as f:
        f.write(f'1 OPENCV {W} {H} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}')
        idx = 1
        for frame in meta['frames']:
            fname = frame['file_path'].split('/')[-1]
            if not (fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.JPG')):
                fname += '.png'
            # blend到opencv的转换：y轴和z轴方向翻转
            pose = np.array(frame['transform_matrix']) @ blender2opencv
            fname2pose.update({fname: pose})
    
    with open('sparse/0/images.txt', 'w') as f:
        for fname in fnames:
            pose = fname2pose[fname]
            # 参考https://blog.csdn.net/weixin_44120025/article/details/124604229：colmap中相机坐标系和世界坐标系是相反的
            # blender中：world = R * camera + T; colmap中：camera = R * world + T
            # 因此转换公式为
            # R’ = R^-1
            # t’ = -R^-1 * t
            R = np.linalg.inv(pose[:3, :3])
            T = -np.matmul(R, pose[:3, 3])
            q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
            q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
            q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
            q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
    
            f.write(f'{idx} {q0} {q1} {q2} {q3} {T[0]} {T[1]} {T[2]} 1 {fname}\n\n')
            idx += 1
    
    with open('sparse/0/points3D.txt', 'w') as f:
       f.write('')

def extract_features(args):
    db=args.colmap_db
    images=args.images
    do_system(f"colmap feature_extractor --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 --database_path {db} --image_path {images}")

def run_colmap(args):
    db=args.colmap_db
    images=args.images
    do_system(f"colmap exhaustive_matcher --database_path {db}")
    do_system(f"colmap point_triangulator --database_path {db} --image_path {images} --input_path sparse/0 --output_path sparse/0")
	
if __name__ == "__main__":
    args = parse_args()
    go_camera2colmap(args)
    print(f"go_camera2colmap finish")
    extract_features(args)
    print(f"extract_features finish")
    transform_colmap_camera.camTodatabase("sparse/0/cameras.txt")
    print(f"transform_colmap_camera finish")
    run_colmap(args)
    print(f"outputting to sparse/0")
	