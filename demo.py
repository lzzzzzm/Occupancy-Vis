import os

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import cv2 as cv
import torch
import argparse
from nuscenes import NuScenes
from visualizer import OccupancyVisualizer
from pyquaternion import Quaternion
import open3d as o3d
import colorsys
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
    parse.add_argument('--vis-scene', type=list, default=['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014'], help='visualize scene list')
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


def change_occupancy_to_bev(occ_semantics, occ_size, free_cls=16):
    # free_cls == 16 as default

    semantics_valid = np.logical_not(occ_semantics == free_cls)
    d = np.arange(occ_size[-1]).reshape(1, 1, occ_size[-1])
    d = np.repeat(d, occ_size[0], axis=0)
    d = np.repeat(d, occ_size[1], axis=1).astype(np.float32)
    d = d * semantics_valid
    selected = np.argmax(d, axis=2)
    selected_torch = torch.from_numpy(selected)
    semantics_torch = torch.from_numpy(occ_semantics)

    occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                 index=selected_torch.unsqueeze(-1))
    occ_bev = occ_bev_torch.numpy()
    occ_bev = occ_bev.flatten().astype(np.int32)
    occ_bev_vis = color_map[occ_bev].astype(np.uint8)
    occ_bev_vis = occ_bev_vis.reshape(occ_size[0], occ_size[1], 3)[::-1, ::-1, :3]
    occ_bev_vis = cv.resize(occ_bev_vis, (occ_size[0], occ_size[1]))

    return occ_bev_vis


def vis_occ_on_bev(vis_scenes_infos, vis_scene, vis_path, pred_path, vis_gt=True):
    for scene_name in vis_scene:
        scene_infos = vis_scenes_infos[scene_name]
        vis_bev_semantics = []
        for info in scene_infos:
            if vis_gt:
                occ_path = info['occ_path']
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
                occ_size=(occ_semantics.shape[0], occ_semantics.shape[1], occ_semantics.shape[2])
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


def vis_occ_on_3d(vis_scenes_infos, vis_scene, vis_path, pred_path, free_cls=16, vis_gt=True, vis_flow=False):
    # check vis path
    mmcv.mkdir_or_exist(vis_path)

    for scene_name in vis_scene:
        scene_infos = vis_scenes_infos[scene_name]
        vis_bev_semantics = []
        buffer_vis_path = '{}/{}'.format(vis_path, scene_name)
        # check vis path
        mmcv.mkdir_or_exist(buffer_vis_path)
        for index, info in enumerate(scene_infos):
            save_path = os.path.join(buffer_vis_path, str(index))

            if vis_gt:
                occ_path = info['occ_path']
                occ_path = occ_path.replace('gts', 'openocc_v2')
                occ_label_path = os.path.join(occ_path, 'labels.npz')
                occ_label = np.load(occ_label_path)
                occ_semantics = occ_label['semantics']
                if vis_flow:
                    occ_flow = occ_label['flow']    # only support for openoccv2
            else:
                token = info['token']
                occ_label_path = os.path.join(pred_path, token + '.npz')
                occ_label = np.load(occ_label_path)
                occ_semantics = occ_label['semantics']
                if vis_flow:
                    occ_flow = occ_label['flow']
            # if view json exits
            occ_visualizer = OccupancyVisualizer(color_map=color_map)
            if os.path.exists('view.json'):
                param = o3d.io.read_pinhole_camera_parameters('view.json')
            else:
                param = None
            occ_visualizer.vis_occ(
                occ_semantics,
                occ_flow,
                ignore_labels=[free_cls],
                voxelSize=(0.4, 0.4, 0.4),
                range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                save_path=save_path,
                wait_time=1,  # 1s, -1 means wait until press q
                view_json=param,
                vis_flow=vis_flow
            )
            # press top-right x to close the windows
            param = occ_visualizer.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters('view.json', param)

            occ_visualizer.o3d_vis.destroy_window()

        # write video
        for i in range(index):
            img_path = os.path.join(buffer_vis_path, str(i) + '.png')
            img = cv.imread(img_path)
            vis_bev_semantics.append(img)
            os.remove(img_path)
        # save video
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        if vis_gt:
            if vis_flow:
                video_path = vis_path + '/' + 'gt-flow_' + scene_name + '.avi'
            else:
                video_path = vis_path + '/' + 'gt-occ_' + scene_name + '.avi'
        else:
            if vis_flow:
                video_path = vis_path + '/' + 'pred-flow_' + scene_name + '.avi'
            else:
                video_path = vis_path + '/' + 'pred-occ_' + scene_name + '.avi'
        video = cv.VideoWriter(video_path, fourcc, 5, (img.shape[1], img.shape[0]))
        for img in vis_bev_semantics:
            video.write(img)
        video.release()
        print('Save video to {}'.format(video_path))


def flow_to_color(vx, vy, max_magnitude=None):
    magnitude = np.sqrt(vx ** 2 + vy ** 2)
    angle = np.arctan2(vy, vx)

    hue = (angle + np.pi) / (2 * np.pi)

    if max_magnitude is None:
        max_magnitude = np.max(magnitude)

    saturation = np.clip(magnitude / max_magnitude, 0, 1)
    value = np.ones_like(saturation)

    hsv = np.stack((hue, saturation, value), axis=-1)
    rgb = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), -1, hsv)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb

def create_legend_circle(radius=1, resolution=500):
    x = np.linspace(-radius, radius, resolution)
    y = np.linspace(-radius, radius, resolution)
    X, Y = np.meshgrid(x, y)
    vx = X
    vy = Y
    magnitude = np.sqrt(vx ** 2 + vy ** 2)
    mask = magnitude <= radius

    vx = vx[mask]
    vy = vy[mask]

    colors = flow_to_color(vx, vy, max_magnitude=radius)

    legend_image = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
    legend_image[mask.reshape(resolution, resolution)] = colors

    return legend_image

if __name__ == '__main__':
    args = parse_args()
    # check vis path
    mmcv.mkdir_or_exist(args.vis_path)

    pkl_file = 'data/nuscenes/bevdetv3-nuscenes_infos_val.pkl'  # generate by mmdet3d
    pkl_data = mmcv.load(pkl_file)
    nusc = NuScenes('v1.0-trainval', args.data_path)
    vis_scenes_infos = arange_according_to_scene(pkl_data['infos'], nusc, args.vis_scene)
    # visualize imgs
    # scene_extrinsic = vis_image(vis_scenes_infos, args.vis_scene, args.vis_path)

    # visualize occupancy on bev plane, gt
    # vis_occ_on_bev(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, vis_gt=True)

    # visualizer occupancy on 3d, gt
    # vis_occ_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path)

    # visualize occupancy on bev plane, pred
    # vis_occ_on_bev(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, vis_gt=False)

    # visualizer occupancy on 3d, pred
    # vis_occ_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, vis_gt=False)

    # visualizer scene flow on 3d, pred
    # vis_occ_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, vis_gt=False, vis_flow=True)
