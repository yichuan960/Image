#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import cv2
import imageio
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import torch
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    segments: dict
    features: list


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, mask_folder , clutter , factor):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        fx, fy, cx, cy = intr.params[0], intr.params[1], intr.params[2], intr.params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        if int(factor) > 1:
            K[:2, :] /= factor
            height = height // factor
            width = width // factor

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        if intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            params = np.array([intr.params[4]], dtype=np.float32)
        elif intr.model == "RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            params = np.array([intr.params[4], intr.params[5], 0.0, 0.0], dtype=np.float32)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            params = np.array([intr.params[4], intr.params[5], intr.params[6], intr.params[7]], dtype=np.float32)
        elif intr.model == "OPENCV_FISHEYE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            params = np.array([intr.params[4], intr.params[5], intr.params[6], intr.params[7]], dtype=np.float32)
        else:
            assert False, "Colmap camera model not handled: only (PINHOLE SIMPLE_PINHOLE SIMPLE_RADIAL RADIAL OPENCV OPENCV_FISHEYE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        #image_dis_path = os.path.join("E:\AI\Robust3DGaussians\data\on-the-go\yoda\distorted", os.path.basename(extr.name))
        # image = imageio.imread(image_path)[..., :3]

        # undistortion
        # params == 0 means no distortion
        # if len(params) > 0:
        #     #print(intr.model)
        #     if extr.name.split('.')[0] == "2clutter1":  # 311 0
        #         pixels = torch.from_numpy(image).float().to("cuda") / 255.0
        #         print(pixels)
        #     K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
        #         K, params, (width, height), 0
        #     )
        #     mapx, mapy = cv2.initUndistortRectifyMap(
        #         K, params, None, K_undist, (width, height), cv2.CV_32FC1
        #     )
        #     # Images are distorted. Undistort them.
        #     image_distorted = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        #     x, y, w, h = roi_undist
        #     image_distorted = image_distorted[y: y + h, x: x + w]
        #     pixels2 = torch.from_numpy(image_distorted).float().to("cuda") / 255.0
        #     canvas2 = (
        #         pixels2
        #         .squeeze(0)
        #         .cpu()
        #         .detach()
        #         .numpy()
        #     )
        #     if extr.name.split('.')[0] == "2clutter1":  # 311 0
        #         pixels2 = torch.from_numpy(image_distorted).float().to("cuda") / 255.0
        #         print(pixels2)
        #     imageio.imwrite(image_dis_path,(canvas2 * 255).astype(np.uint8),)

        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        with open(os.path.join(mask_folder, os.path.basename(extr.name).split('.')[0] + '.json'), 'r') as file:
            segments = json.load(file)

        # Get SD features
        features = []
        load_keyword = "clutter"

        image_id = extr.name.split(".")[-2]
        sd_folder = images_folder.rsplit(os.sep,1)[0]
        if extr.name.find(load_keyword) != -1:
            feature_path = os.path.join(
                os.path.join(sd_folder, "SD"), f"{image_id}.npy"
            )
            feature = np.load(feature_path)
            if clutter:
                ft_flat = np.transpose(feature.reshape((1280, 50 * 50)), (1, 0))
                x = np.linspace(0, 1, 50)
                y = np.linspace(0, 1, 50)
                xv, yv = np.meshgrid(x, y)
                indxy = np.reshape(np.stack([xv, yv], axis=-1), (50 * 50, 2))
                knn_graph = kneighbors_graph(indxy, 8, include_self=False)
                model = AgglomerativeClustering(
                    linkage="ward", connectivity=knn_graph, n_clusters=100
                )
                model.fit(ft_flat)
                feature = np.array(
                    [model.labels_ == i for i in range(model.n_clusters)],
                    dtype=np.float32,
                ).reshape((model.n_clusters, 50, 50))
            features.append(feature)
            # if extr.name == "2clutter000.JPG":
            #     print(feature.shape)
            #     print(np.nonzero(feature))
        else:
            features.append([])

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              segments=segments,features=features)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, config):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    mask_dir = os.path.join(path, 'segments', 'masks')
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir), mask_folder=mask_dir, clutter=config['clutter'], factor=config['factor'])
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        if config['train_keyword'] == "":
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx >= config['test_size']]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx < config['test_size']]
        else:
            train_cam_infos = []
            test_cam_infos = []
            for idx, c in enumerate(cam_infos):
                if c.image_name.find(config['train_keyword']) != -1:
                    train_cam_infos.append(c)
                else:
                    test_cam_infos.append(c)
            #train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name.find(config['train_keyword'])]
            #test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name.find(config['test_keyword'])]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    print(len(train_cam_infos))
    print(len(test_cam_infos))
    # Only load 1 camera for debugging
    if "debug" in config and config["debug"]:
        if len(test_cam_infos) >= 1:
            test_cam_infos = test_cam_infos[:1]
        train_cam_infos = train_cam_infos[:1]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}
