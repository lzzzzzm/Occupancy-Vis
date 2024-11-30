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
        [255, 120, 50],     # barrier              orange
        [255, 192, 203],    # bicycle              pink
        [255, 255, 0],      # bus                  yellow
        [0, 150, 245],      # car                  blue
        [0, 255, 255],      # construction_vehicle cyan
        [255, 127, 0],      # motorcycle           dark orange
        [255, 0, 0],        # pedestrian           red
        [255, 240, 150],    # traffic_cone         light yellow
        [135, 60, 0],       # trailer              brown
        [160, 32, 240],     # truck                purple
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dard purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255]     # free                 white
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

def vis_lidar_on_3d(vis_scenes_infos, vis_scene, vis_path):
    save_imgs = []
    mmcv.mkdir_or_exist(vis_path)
    for scene_name in vis_scene:
        buffer_vis_path = '{}/{}'.format(vis_path, scene_name)
        mmcv.mkdir_or_exist(buffer_vis_path)    # check vis path
        scene_infos = vis_scenes_infos[scene_name]
        vis_lidar_points = []
        for index, info in enumerate(scene_infos):
            if os.path.exists('view.json'):
                param = o3d.io.read_pinhole_camera_parameters('view.json')
            else:
                param = None

            save_path = os.path.join(buffer_vis_path, str(index))
            lidar_points = np.fromfile(info['lidar_path'], dtype=np.float32).reshape(-1, 5)[:, :3]

            lidar_vis = LidarVisualizer(color_map=color_map)
            ego_lidar_vis = LidarVisualizer(color_map=color_map)
            lidar_vis.vis_lidar_points(
                lidar_points,
                save_path=save_path,
                view_json=param,
            )

            param = lidar_vis.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters('view.json', param)

            lidar_vis.o3d_vis.destroy_window()

        # write video
        for i in range(index):
            img_path = os.path.join(buffer_vis_path, str(i) + '.png')
            img = cv.imread(img_path)
            vis_lidar_points.append(img)
            os.remove(img_path)

        # wirte to video
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        video_path = 'demo_out/' + 'liadr_' + scene_name + '.avi'
        video = cv.VideoWriter(video_path, fourcc, 5, (img.shape[1], img.shape[0]))
        for img in vis_lidar_points:
            video.write(img)
        video.release()
        print('Save video to {}'.format(video_path))


if __name__ == '__main__':
    print('open3d version: {}, if you want to use viewcontrol, make sure using 0.16.0 version!!'.format(o3d.__version__))
    args = parse_args()
    # check vis path
    mmcv.mkdir_or_exist(args.vis_path)

    pkl_file = 'data/nuscenes/nus-infos/bevdetv3-nuscenes_infos_val.pkl'  # generate by mmdet3d
    pkl_data = mmcv.load(pkl_file)
    nusc = NuScenes('v1.0-trainval', args.data_path)
    vis_scenes_infos = arange_according_to_scene(pkl_data['infos'], nusc, args.vis_scene)

    vis_lidar_on_3d(vis_scenes_infos, args.vis_scene, vis_path=args.vis_path)

