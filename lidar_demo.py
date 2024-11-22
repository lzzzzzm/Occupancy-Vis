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

def sample_augmentation(H, W, data_config):
    fH, fW = data_config['input_size']

    resize = float(fW) / float(W)
    resize += np.random.uniform(*data_config['resize'])
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    random_crop_height = \
        data_config.get('random_crop_height', False)
    if random_crop_height:
        crop_h = int(np.random.uniform(max(0.3*newH, newH-fH),
                                       newH-fH))
    else:
        crop_h = \
            int((1 - np.random.uniform(*data_config['crop_h'])) *
                 newH) - fH
    crop_w = int(np.random.uniform(0, max(0, newW - fW)))
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = data_config['flip'] and np.random.choice([0, 1])
    rotate = np.random.uniform(*data_config['rot'])
    if data_config.get('vflip', False) and np.random.choice([0, 1]):
        rotate += 180

    return resize, resize_dims, crop, flip, rotate

def get_rot(theta):
    return torch.Tensor([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ])

def img_transform_core(img, resize_dims, crop, flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)
    return img

def img_transform_core_opencv(img, post_rot, post_tran,
                              crop):
    img = np.array(img).astype(np.float32)
    img = cv.warpAffine(img,
                         np.concatenate([post_rot,
                                        post_tran.reshape(2,1)],
                                        axis=1),
                         (crop[2]-crop[0], crop[3]-crop[1]),
                         flags=cv.INTER_LINEAR)
    return img

def img_transform(img, post_rot, post_tran, resize, resize_dims,
                  crop, flip, rotate):
    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b
    img = img_transform_core_opencv(img, post_rot, post_tran, crop)
    copy_img = img.copy()
    invalid_index = np.where(np.array(copy_img)==0)

    return img, post_rot, post_tran, invalid_index

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
                img_augs = sample_augmentation(h, w, data_config)
                resize, resize_dims, crop, flip, rotate = img_augs
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)
                img, post_rot, post_tran, invalid_index = img_transform(
                    img,
                    post_rot,
                    post_tran,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate)
                post_augs = torch.eye(4)
                post_augs[:2, :2] = post_rot
                post_augs[:2, 2] = post_tran
                h, w, _ = img.shape

                cam2img = np.eye(4)
                cam2img[:3, :3] = cam_info['cam_intrinsic']
                cam2img = torch.from_numpy(cam2img).to(torch.float32)
                # update cam2img
                cam2img = post_augs @ cam2img

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

def map_to_bev(lidar_points):
    bev_w = 200
    bev_h = 200
    res = 0.4
    bev_image = np.zeros((bev_h, bev_w, 3))
    point_cloud_range = [-40.0, -40.0, 40.0, 40.0]
    max_distance = np.sqrt(np.max(lidar_points[:, 0] ** 2 + lidar_points[:, 1] ** 2))
    # 将点云坐标转换为BEV坐标
    for point in lidar_points:
        x, y, z = point
        if x < point_cloud_range[0] or x >= point_cloud_range[2] or y < point_cloud_range[1] or y >= point_cloud_range[3]:
            continue
        # 转换为BEV图像坐标
        bev_x = x / res + bev_w / 2
        bev_y = y / res + bev_h / 2
        distance = np.sqrt(x ** 2 + y ** 2) * 10
        bev_x = int(bev_x)
        bev_y = int(bev_y)

        if 0 <= bev_x < bev_w and 0 <= bev_y < bev_h:
            # 根据距离上色，这里使用简单的线性映射，您可以根据需要调整颜色映射函数
            color_value = 177
            # 将颜色值应用到BEV图像的RGB通道
            bev_image[bev_y, bev_x] = [color_value, color_value, color_value]  # 灰度值，可以根据需要调整为RGB颜色

    # change a-xis to y-axis
    bev_image = np.transpose(bev_image, (1, 0, 2))
    bev_image = bev_image.astype(np.uint8)

    return bev_image


def vis_lidar_on_3d(vis_scenes_infos, vis_scene, map_bev=False):
    bev_w = 200
    bev_h = 200
    res = 0.4
    save_imgs = []
    for scene_name in vis_scene:
        scene_infos = vis_scenes_infos[scene_name]
        vis_lidar_points = []
        for info in scene_infos:
            lidar_points = np.fromfile(info['lidar_path'], dtype=np.float32).reshape(-1, 5)[:, :3]
            # change x-y axis
            lidar_points[:, 0], lidar_points[:, 1] = lidar_points[:, 1], -lidar_points[:, 0]
            if map_bev:
                bev_image = map_to_bev(lidar_points)
                cv2.namedWindow('bev', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('bev', 400, 400)
                cv2.imshow('bev', bev_image)
                cv2.waitKey(0)

                save_imgs.append(bev_image)
            else:
                vis_lidar_points.append(lidar_points)

                lidar_vis = LidarVisualizer(color_map=color_map)
                lidar_vis.vis_lidar_points(lidar_points)
                lidar_vis.o3d_vis.destroy_window()
    # wirte to video
    if map_bev:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        video_path = 'demo_out/' + 'bev_' + scene_name + '.avi'
        video = cv.VideoWriter(video_path, fourcc, 5, (bev_image.shape[0], bev_image.shape[1]))
        for img in save_imgs:
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

    vis_lidar_on_3d(vis_scenes_infos, args.vis_scene, map_bev=True)

    # vis_lidar_on_img(vis_scenes_infos, args.vis_scene)

