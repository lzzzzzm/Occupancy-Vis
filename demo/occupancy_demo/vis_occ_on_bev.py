import os

import cv2
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
        [0, 0, 0],          # unlabeled            black
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
openocc_colors_map = np.array(
    [
        [0, 150, 245],      # car                  blue         √
        [160, 32, 240],     # truck                purple       √
        [135, 60, 0],       # trailer              brown        √
        [255, 255, 0],      # bus                  yellow       √
        [0, 255, 255],      # construction_vehicle cyan         √
        [255, 192, 203],    # bicycle              pink         √
        [255, 127, 0],      # motorcycle           dark orange  √
        [255, 0, 0],        # pedestrian           red          √
        [255, 240, 150],    # traffic_cone         light yellow
        [255, 120, 50],     # barrier              orange
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dard purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255],    # Free                 White
    ]
)

foreground_idx = [0, 1, 2, 3, 4, 5, 6, 7]


def parse_args():
    parse = argparse.ArgumentParser('')
    parse.add_argument('--pkl-file', type=str, default='data/nuscenes/nus-infos/bevdetv3-nuscenes_infos_val.pkl', help='path of pkl for the nuScenes dataset')
    parse.add_argument('--data-path', type=str, default='data/nuscenes', help='path of the nuScenes dataset')
    parse.add_argument('--data-version', type=str, default='v1.0-trainval', help='version of the nuScenes dataset')
    parse.add_argument('--dataset-type', type=str, default='occ3d', help='version of the nuScenes dataset')
    parse.add_argument('--pred-path', type=str, default='scene-0274', help='version of the nuScenes dataset')
    parse.add_argument('--vis-scene', type=list, default=['scene-0274'], help='visualize scene list')
    parse.add_argument('--vis-path', type=str, default='demo_out', help='path of saving the visualization images')
    parse.add_argument('--vis-single-data', type=str, default='scene-0277/pred/ef710f7aad4c4bcf9ac21ef155c8c3d1.npz', help='single path of the visualization data')
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

def change_occupancy_to_bev(occ_semantics, occ_size, free_cls=16, colors=None):
    # free_cls == 16 as default

    semantics_valid = np.logical_not(occ_semantics == free_cls)
    d = np.arange(occ_size[-1]).reshape(1, 1, occ_size[-1])
    d = np.repeat(d, occ_size[0], axis=0)
    d = np.repeat(d, occ_size[1], axis=1).astype(np.float32)
    d = d * semantics_valid
    selected = np.argmax(d, axis=2)
    selected_torch = torch.from_numpy(selected)
    semantics_torch = torch.from_numpy(occ_semantics)

    occ_bev_torch = torch.gather(semantics_torch, dim=2, index=selected_torch.unsqueeze(-1))
    occ_bev = occ_bev_torch.numpy()
    occ_bev = occ_bev.flatten().astype(np.int32)
    occ_bev_vis = colors[occ_bev].astype(np.uint8)
    occ_bev_vis = occ_bev_vis.reshape(occ_size[0], occ_size[1], 3)[::-1, ::-1, :3]
    occ_bev_vis = cv.resize(occ_bev_vis, (occ_size[0], occ_size[1]))
    occ_bev_vis = cv.cvtColor(occ_bev_vis, cv.COLOR_RGB2BGR)

    return occ_bev_vis


def vis_scene_occ_on_bev(vis_scenes_infos, vis_path, pred_path, dataset_type, vis_gt=True):

    for scene_name in vis_scenes_infos:
        scene_infos = vis_scenes_infos[scene_name]
        vis_bev_semantics = []
        for info in scene_infos:
            if vis_gt:
                occ_path = info['occ_path']
                if dataset_type == 'openocc':
                    occ_path = occ_path.replace('gts', 'openocc_v2')
                occ_label_path = os.path.join(occ_path, 'labels.npz')
                occ_label = np.load(occ_label_path)
                occ_semantics = occ_label['semantics']
            else:
                token = info['token']
                occ_label_path = os.path.join(pred_path, token+'.npz')
                occ_label = np.load(occ_label_path)
                occ_semantics = occ_label['semantics']

            bev_semantics = change_occupancy_to_bev(
                occ_semantics,
                occ_size=(occ_semantics.shape[0], occ_semantics.shape[1], occ_semantics.shape[2]),
                free_cls=16 if dataset_type=='openocc' else 17,
                colors=openocc_colors_map if dataset_type=='openocc' else color_map
            )
            vis_bev_semantics.append(bev_semantics)

        # save video
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        if vis_gt:
            video_path = vis_path + '/' + 'gt-occ-bev_' + scene_name + '.avi'
        else:
            video_path = vis_path + '/' + 'pred-occ-bev_' + scene_name + '.avi'
        video = cv.VideoWriter(video_path, fourcc, 5, (bev_semantics.shape[0], bev_semantics.shape[1]))
        for img in vis_bev_semantics:
            video.write(img)
        video.release()
        print('Save video to {}'.format(video_path))

def vis_single_occ_on_bev(data_path, dataset_type):
    occ_label = np.load(data_path)
    occ_semantics = occ_label['semantics']
    bev_semantics = change_occupancy_to_bev(
        occ_semantics,
        occ_size=(occ_semantics.shape[0], occ_semantics.shape[1], occ_semantics.shape[2]),
        free_cls=16 if dataset_type=='openocc' else 17,
        colors=openocc_colors_map if dataset_type=='openocc' else color_map
    )
    cv.imshow('bev_semantics', bev_semantics)
    cv.waitKey()

def vis_forecast_occ_on_bev(data_path, dataset_type):
    occ_label = np.load(data_path)
    occ_semantics = occ_label['semantics']

    vis_semantics = []
    for semantics in occ_semantics:
        bev_semantics = change_occupancy_to_bev(
            semantics,
            occ_size=(semantics.shape[0], semantics.shape[1], semantics.shape[2]),
            free_cls=16 if dataset_type=='openocc' else 17,
            colors=openocc_colors_map if dataset_type=='openocc' else color_map
        )
        bev_semantics = cv.cvtColor(bev_semantics, cv.COLOR_BGR2RGB)
        vis_semantics.append(bev_semantics)


    for i, bev_semantics in enumerate(vis_semantics):
        # plt.subplot(1, len(vis_semantics), i + 1)
        plt.figure()
        plt.imshow(bev_semantics)
        plt.axis('off')
        plt.savefig('forecast_occ_bev_{}.png'.format(i), bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.savefig('forecast_occ_bev.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

if __name__ == '__main__':
    print('open3d version: {}, if you want to use viewcontrol, make sure using 0.16.0 version!!'.format(o3d.__version__))
    args = parse_args()
    # check vis path
    mmcv.mkdir_or_exist(args.vis_path)
    #
    # pkl_data = mmcv.load(args.pkl_file)
    # nusc = NuScenes(args.data_version, args.data_path)
    # vis_scenes_infos = arange_according_to_scene(pkl_data['infos'], nusc, args.vis_scene)
    # # GT visualization
    # vis_scene_occ_on_bev(vis_scenes_infos, args.vis_path,  args.pred_path, args.dataset_type, vis_gt=True)
    # # Prediction visualization
    # vis_scene_occ_on_bev(vis_scenes_infos, args.vis_path,  args.pred_path, args.dataset_type, vis_gt=False)
    # # Visualize single data
    # vis_single_occ_on_bev(args.vis_single_data, args.dataset_type)
    # Visualize forecast data
    vis_forecast_occ_on_bev(args.vis_single_data, args.dataset_type)
