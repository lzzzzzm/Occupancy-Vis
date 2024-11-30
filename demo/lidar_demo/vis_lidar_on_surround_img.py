import os

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import cv2 as cv
import torch
import argparse
from nuscenes import NuScenes
from visualizer import LidarVisualizer
from pyquaternion import Quaternion
import open3d as o3d
import colorsys
from PIL import Image
from nuscenes.utils.geometry_utils import transform_matrix

color_map = np.array(
    [
        [255, 120, 50],  # barrier              orange
        [255, 192, 203],  # bicycle              pink
        [255, 255, 0],  # bus                  yellow
        [0, 150, 245],  # car                  blue
        [0, 255, 255],  # construction_vehicle cyan
        [255, 127, 0],  # motorcycle           dark orange
        [255, 0, 0],  # pedestrian           red
        [255, 240, 150],  # traffic_cone         light yellow
        [135, 60, 0],  # trailer              brown
        [160, 32, 240],  # truck                purple
        [255, 0, 255],  # driveable_surface    dark pink
        [139, 137, 137],  # other_flat           dark red
        [75, 0, 75],  # sidewalk             dard purple
        [150, 240, 80],  # terrain              light green
        [230, 230, 250],  # manmade              white
        [0, 175, 0],  # vegetation           green
        [255, 255, 255]  # free                 white
    ]
)


def parse_args():
    parse = argparse.ArgumentParser('')
    parse.add_argument('--data-path', type=str, default='data/nuscenes', help='path of the nuScenes dataset')
    parse.add_argument('--pred-path', type=str, default='predictions', help='path of the prediction data')
    parse.add_argument('--vis-scene', type=list, default=['scene-0922'], help='visualize scene list')
    parse.add_argument('--vis-path', type=str, default='demo_out', help='path of saving the visualization images')
    parse.add_argument('--single-data-path', type=str, default=None, help='single data path for visualization')
    parse.add_argument('--car-model-data-path', type=str, default=None, help='car model for visualization')
    parse.add_argument('--dataset_type', type=str, default='occ3d', help='dataset type')
    args = parse.parse_args()
    return args


def arange_according_to_scene(infos, nusc, vis_scene):
    scenes = dict()

    for i, info in enumerate(infos):
        scene_token = nusc.get('sample', info['token'])['scene_token']
        scene_meta = nusc.get('scene', scene_token)
        scene_name = scene_meta['name']
        if not scene_name in scenes:
            scenes[scene_name] = [info]
        else:
            scenes[scene_name].append(info)

    vis_scenes = dict()
    for scene_name in vis_scene:
        vis_scenes[scene_name] = scenes[scene_name]

    return vis_scenes


def points2depthmap(points, height, width, grid_config):
    depth_map = torch.zeros((height, width), dtype=torch.float32)
    coor = torch.round(points[:, :2])
    depth = points[:, 2]
    kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
        coor[:, 1] >= 0) & (coor[:, 1] < height) & (
            depth < grid_config['depth'][1]) & (
                depth >= grid_config['depth'][0])
    coor, depth = coor[kept1], depth[kept1]
    ranks = coor[:, 0] + coor[:, 1] * width
    sort = (ranks + depth / 100.).argsort()
    coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

    kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
    kept2[1:] = (ranks[1:] != ranks[:-1])
    coor, depth = coor[kept2], depth[kept2]

    coor = coor.to(torch.long)
    depth_map[coor[:, 1], coor[:, 0]] = depth
    return depth_map

def vis_depth(img, depth):
    depth = depth.cpu().numpy()
    # img = img.permute(1, 2, 0).cpu().numpy()
    # img = img * std + mean
    img = np.array(img, dtype=np.uint8)
    invalid_y, invalid_x, invalid_c = np.where(img==0)
    depth[invalid_y, invalid_x] = 0
    y, x = np.where(depth!=0)
    plt.figure()
    plt.imshow(img)
    plt.scatter(x, y, c=depth[y, x], cmap='rainbow_r', alpha=0.5, s=2)
    plt.show()

def vis_lidar_on_img(vis_scenes_infos, vis_scene):
    grid_config = {
        'depth': [1.0, 52.0, 0.5],
    }
    cam_info_name = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    data_config = {
        'cams': cam_info_name,
        'Ncams': 6,
        'input_size': (256, 704),
        'src_size': (900, 1600),
        # Augmentation
        'resize': (-0.06, 0.11),
        'rot': (-5.4, 5.4),
        'flip': True,
        'crop_h': (0.0, 0.0),
        'resize_test': 0.00,
    }
    for scene_name in vis_scene:
        scene_infos = vis_scenes_infos[scene_name]

        for info in scene_infos:
            lidar_points = np.fromfile(info['lidar_path'], dtype=np.float32).reshape(-1, 5)[:, :3]
            lidar_points = torch.from_numpy(lidar_points)

            # get lidar2ego
            lidar2lidarego = transform_matrix(
                translation=info['lidar2ego_translation'],
                rotation=Quaternion(info['lidar2ego_rotation'])
            )
            lidar2lidarego = torch.from_numpy(lidar2lidarego).to(torch.float32)

            # get lidarego2global
            lidarego2global = transform_matrix(
                translation=info['ego2global_translation'],
                rotation=Quaternion(info['ego2global_rotation'])
            )
            lidarego2global = torch.from_numpy(lidarego2global).to(torch.float32)

            cams_infos = info['cams']
            for cam_name in cam_info_name:
                cam_info = cams_infos[cam_name]
                cam_path = cam_info['data_path']

                img = cv.imread(cam_path)
                h, w, _ = img.shape

                cam2img = np.eye(4)
                cam2img[:3, :3] = cam_info['cam_intrinsic']
                cam2img = torch.from_numpy(cam2img).to(torch.float32)

                # get camego2cam
                camego2cam = transform_matrix(
                    translation=cam_info['sensor2ego_translation'],
                    rotation=Quaternion(cam_info['sensor2ego_rotation']),
                    inverse=True
                )
                camego2cam = torch.from_numpy(camego2cam).to(torch.float32)

                # get global2camego
                global2camego = transform_matrix(
                    translation=cam_info['ego2global_translation'],
                    rotation=Quaternion(cam_info['ego2global_rotation']),
                    inverse=True
                )
                global2camego = torch.from_numpy(global2camego).to(torch.float32)

                # get lidar2img
                lidar2img = cam2img @ camego2cam @ global2camego @ lidarego2global @ lidar2lidarego

                # project lidar points to image
                img_points = lidar2img @ torch.cat([lidar_points.T, torch.ones((1, lidar_points.shape[0]))], dim=0)
                img_points = img_points.permute(1, 0)
                img_points = torch.cat([img_points[:, :2] / img_points[:, 2].unsqueeze(1), img_points[:, 2].unsqueeze(1)], dim=1)

                depth_map = points2depthmap(img_points, h, w, grid_config)

                vis_depth(img, depth_map)

if __name__ == '__main__':
    print(
        'open3d version: {}, if you want to use viewcontrol, make sure using 0.16.0 version!!'.format(o3d.__version__))
    args = parse_args()
    # check vis path
    mmcv.mkdir_or_exist(args.vis_path)

    pkl_file = 'data/nuscenes/nus-infos/bevdetv3-nuscenes_infos_val.pkl'  # generate by mmdet3d
    pkl_data = mmcv.load(pkl_file)
    nusc = NuScenes('v1.0-trainval', args.data_path)
    vis_scenes_infos = arange_according_to_scene(pkl_data['infos'], nusc, args.vis_scene)

    vis_lidar_on_img(vis_scenes_infos, args.vis_scene)
