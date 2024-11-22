import os

import matplotlib.pyplot as plt
import mmcv
import copy
import numpy as np
import cv2 as cv
import torch
import argparse
from nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from visualizer import OccupancyVisualizer
from pyquaternion import Quaternion
import open3d as o3d
import colorsys
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

foreground_idx = [0, 1, 2, 3, 4, 5, 6, 7]


def parse_args():
    parse = argparse.ArgumentParser('')
    parse.add_argument('--pkl-file', type=str, default='data/nuscenes/nus-infos/bevdetv3-nuscenes_infos_val.pkl', help='path of pkl for the nuScenes dataset')
    parse.add_argument('--data-path', type=str, default='data/nuscenes', help='path of the nuScenes dataset')
    parse.add_argument('--data-version', type=str, default='v1.0-trainval', help='version of the nuScenes dataset')
    parse.add_argument('--vis-scene', type=list, default=['scene-0274'], help='visualize scene list')
    parse.add_argument('--vis-path', type=str, default='demo_out', help='path of saving the visualization images')
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
    if len(vis_scene) == 0:
        vis_scenes = scenes
    else:
        for scene_name in vis_scene:
            vis_scenes[scene_name] = scenes[scene_name]

    return vis_scenes


def vis_image(vis_scenes_infos, vis_scene, vis_path, load_cam_info=False):
    cam_info_name = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    scene_extrinsic = dict()
    for scene_name in vis_scene:
        scene_infos = vis_scenes_infos[scene_name]
        extrinsic_list = []
        vis_imgs = []
        for info in scene_infos:
            cams_infos = info['cams']
            imgs = []
            cam_extrinsic = dict()
            for cam_name in cam_info_name:
                cam_info = cams_infos[cam_name]
                cam_path = cam_info['data_path']
                if load_cam_info:
                    sensor2ego_rots = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
                    sensor2ego_trans = cam_info['sensor2ego_translation']
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = sensor2ego_rots
                    extrinsic[:3, 3] = sensor2ego_trans
                    cam_extrinsic[cam_name] = extrinsic
                img = cv.imread(cam_path)
                imgs.append(img)
            # save extrinsic
            if load_cam_info:
                extrinsic_list.append(cam_extrinsic)
            # concate imgs 2x3
            img_front = cv.hconcat([imgs[1], imgs[0], imgs[2]])
            img_back = cv.hconcat([imgs[4], imgs[3], imgs[5]])
            vis_img = cv.vconcat([img_front, img_back])
            vis_img = cv.resize(vis_img, (1920, 1080))
            vis_imgs.append(vis_img)

        # save video
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        video_path = vis_path + '/' + 'surround-view_' + scene_name + '.avi'
        video = cv.VideoWriter(video_path, fourcc, 5, (1920, 1080))
        for img in vis_imgs:
            video.write(img)
        video.release()
        scene_extrinsic[scene_name] = extrinsic_list
        print('Save video to {}'.format(video_path))
    return scene_extrinsic



if __name__ == '__main__':
    print('open3d version: {}, if you want to use viewcontrol, make sure using 0.16.0 version!!'.format(o3d.__version__))
    args = parse_args()
    # check vis path
    mmcv.mkdir_or_exist(args.vis_path)

    pkl_data = mmcv.load(args.pkl_file)
    nusc = NuScenes(args.data_version, args.data_path)
    vis_scenes_infos = arange_according_to_scene(pkl_data['infos'], nusc, args.vis_scene)
    # visualize imgs
    scene_extrinsic = vis_image(vis_scenes_infos, args.vis_scene, args.vis_path)
