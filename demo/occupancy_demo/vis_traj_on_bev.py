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
from nuscenes.utils.geometry_utils import transform_matrix
from visualizer import OccupancyVisualizer
from pyquaternion import Quaternion
import open3d as o3d
import colorsys
from skimage.draw import polygon
from scipy.interpolate import make_interp_spline

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

binary_colors_map = np.array(
    [
        [0, 0, 0],
        [255, 255, 255]
    ]
)

foreground_idx = [0, 1, 2, 3, 4, 5, 6, 7]


def parse_args():
    parse = argparse.ArgumentParser('')
    parse.add_argument('--pkl-file', type=str, default='data/nuscenes/world-nuscenes_mini_infos_val.pkl', help='path of pkl for the nuScenes dataset')
    parse.add_argument('--data-path', type=str, default='data/nuscenes', help='path of the nuScenes dataset')
    parse.add_argument('--data-version', type=str, default='v1.0-mini', help='version of the nuScenes dataset')
    parse.add_argument('--dataset-type', type=str, default='openocc', help='version of the nuScenes dataset')
    parse.add_argument('--pred-path', type=str, default='scene-0274', help='version of the nuScenes dataset')
    parse.add_argument('--vis-scene', type=list, default=['scene-0916'], help='visualize scene list')
    parse.add_argument('--vis-path', type=str, default='demo_out', help='path of saving the visualization images')
    parse.add_argument('--vis-single-data', type=str, default='scene-0274/1fa5506ca31d4174955140d2138db679.npz', help='single path of the visualization data')
    args = parse.parse_args()
    return args


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = np.array([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = np.array([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = np.array([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]], dtype=np.int64)

    return bev_resolution, bev_start_position, bev_dimension


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

def change_occupancy_to_bev(occ_semantics, occ_size, free_cls=16, colors=None, binary=False):
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

    if binary:
        occ_bev_binary = np.zeros_like(occ_bev)
        occ_bev_binary[occ_bev==free_cls] = 0
        occ_bev_binary[occ_bev!=free_cls] = 1
        occ_bev_vis = binary_colors_map[occ_bev_binary].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(occ_size[0], occ_size[1], 3)[::-1, ::-1, :3]
        occ_bev_vis = cv.resize(occ_bev_vis, (occ_size[0], occ_size[1]))
        occ_bev_vis = cv.cvtColor(occ_bev_vis, cv.COLOR_RGB2BGR)
    else:
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

            # cv2.imshow('bev_semantics', bev_semantics)
            # cv2.waitKey()

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

def decode_pose(pose_mat, ego_to_globals, ego_to_lidar=None):
    bs, f = pose_mat.shape[:2]
    # check type to numpy
    if isinstance(ego_to_globals, torch.Tensor):
        ego_to_globals = ego_to_globals.cpu().numpy()
    if isinstance(pose_mat, torch.Tensor):
        pose_mat = pose_mat.cpu().numpy()
    if ego_to_lidar is not None:
        if isinstance(ego_to_lidar, torch.Tensor):
            ego_to_lidar = ego_to_lidar.cpu().numpy()
    global_to_ego = np.linalg.inv(ego_to_globals)

    outs = []
    for i in range(bs):
        trajs = []
        for j in range(f - 1):
            curr_xyz = pose_mat[i, j, :2, 3]
            next_xyz = pose_mat[i, j + 1, :2, 3]
            global_pose = np.array([curr_xyz, next_xyz])
            global_pose = np.concatenate([global_pose, np.zeros((global_pose.shape[0], 1))], axis=1)
            # trans to ego
            global_pose = np.concatenate([global_pose, np.ones((global_pose.shape[0], 1))], axis=1)
            global_pose = np.dot(global_to_ego[i, j], global_pose.T).T

            # trans to lidar
            if ego_to_lidar is not None:
                global_pose = np.dot(ego_to_lidar[i, j], global_pose.T).T

            # get trajs
            ego_trajs = global_pose[1:] - global_pose[:-1]
            trajs.append(ego_trajs[:, :2].squeeze())
        outs.append(trajs)
    return np.array(outs)

def _get_poly_region_in_image(param, bev_resolution, bev_start_position):
    lidar2cv_rot = np.array([[1, 0], [0, -1]])
    x_a, y_a, yaw_a, agent_length, agent_width = param
    trans_a = np.array([[x_a, y_a]]).T
    rot_mat_a = np.array([[np.cos(yaw_a), -np.sin(yaw_a)],
                          [np.sin(yaw_a), np.cos(yaw_a)]])
    agent_corner = np.array([
        [agent_length / 2, -agent_length / 2, -agent_length / 2, agent_length / 2],
        [agent_width / 2, agent_width / 2, -agent_width / 2, -agent_width / 2]])  # (2,4)
    agent_corner_lidar = np.matmul(rot_mat_a, agent_corner) + trans_a  # (2,4)
    # convert to cv frame
    agent_corner_cv2 = (np.matmul(lidar2cv_rot, agent_corner_lidar) \
                        - bev_start_position[:2, None] + bev_resolution[:2, None] / 2.0).T / bev_resolution[:2]  # (4,2)
    agent_corner_cv2 = np.round(agent_corner_cv2).astype(np.int32)

    return agent_corner_cv2

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

def get_birds_eye_view_label(
        gt_agent_boxes,
        gt_agent_feats
):
    '''
    gt_agent_boxes (LiDARInstance3DBoxes): list of GT Bboxs.
        dim 9 = (x,y,z)+(w,l,h)+yaw+(vx,vy)
    gt_agent_feats: (B, A, 34)
        dim 34 = fut_traj(6*2) + fut_mask(6) + goal(1) + lcf_feat(9) + fut_yaw(6)
        lcf_feat (x, y, yaw, vx, vy, width, length, height, type)
    ego_lcf_feats: (B, 9)
        dim 8 = (vx, vy, ax, ay, w, length, width, vel, steer)
    '''
    T = 6
    category_index_map = {
        'human': [2, 3, 4, 5, 6, 7, 8],
        'vehicle': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    }
    X_BOUND = [-50.0, 50.0, 0.5]  # Forward
    Y_BOUND = [-50.0, 50.0, 0.5]  # Sides
    Z_BOUND = [-10.0, 10.0, 20.0]  # Height
    bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(X_BOUND, Y_BOUND, Z_BOUND)

    segmentation = np.zeros((T, bev_dimension[0], bev_dimension[1]))
    pedestrian = np.zeros((T, bev_dimension[0], bev_dimension[1]))
    agent_num = gt_agent_feats.shape[1]

    gt_agent_fut_trajs = gt_agent_feats[..., :T * 2].reshape(-1, 6, 2)
    gt_agent_fut_mask = gt_agent_feats[..., T * 2:T * 3].reshape(-1, 6)
    # gt_agent_lcf_feat = gt_agent_feats[..., T*3+1:T*3+10].reshape(-1, 9)
    gt_agent_fut_yaw = gt_agent_feats[..., T * 3 + 10:T * 4 + 10].reshape(-1, 6, 1)
    gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
    gt_agent_fut_yaw = np.cumsum(gt_agent_fut_yaw, axis=1)

    gt_agent_boxes[:, 6:7] = -1 * (gt_agent_boxes[:, 6:7] + np.pi / 2)  # NOTE: convert yaw to lidar frame
    gt_agent_fut_trajs = gt_agent_fut_trajs + gt_agent_boxes[:, np.newaxis, 0:2]
    gt_agent_fut_yaw = gt_agent_fut_yaw + gt_agent_boxes[:, np.newaxis, 6:7]

    for t in range(T):
        for i in range(agent_num):
            if gt_agent_fut_mask[i][t] == 1:
                # Filter out all non vehicle instances
                category_index = int(gt_agent_feats[0, i][27])
                agent_length, agent_width = gt_agent_boxes[i][4], gt_agent_boxes[i][3]
                x_a = gt_agent_fut_trajs[i, t, 0]
                y_a = gt_agent_fut_trajs[i, t, 1]
                yaw_a = gt_agent_fut_yaw[i, t, 0]
                param = [x_a, y_a, yaw_a, agent_length, agent_width]
                if (category_index in category_index_map['vehicle']):
                    poly_region = _get_poly_region_in_image(param, bev_resolution, bev_start_position)
                    cv2.fillPoly(segmentation[t], [poly_region], 1.0)
                if (category_index in category_index_map['human']):
                    poly_region = _get_poly_region_in_image(param, bev_resolution, bev_start_position)
                    cv2.fillPoly(pedestrian[t], [poly_region], 1.0)

    return segmentation, pedestrian

def get_box_info(info):
    # Load Box info
    mask = info['valid_flag']
    gt_bboxes_3d = info['gt_boxes'][mask]
    gt_names_3d = info['gt_names'][mask]
    gt_velocity = info['gt_velocity'][mask]
    nan_mask = np.isnan(gt_velocity[:, 0])
    gt_velocity[nan_mask] = [0.0, 0.0]
    gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

    gt_fut_trajs = info['gt_agent_fut_trajs'][mask]  # N, 2*6
    gt_fut_masks = info['gt_agent_fut_masks'][mask]  # N, 6
    gt_fut_goal = info['gt_agent_fut_goal'][mask]    # N
    gt_lcf_feat = info['gt_agent_lcf_feat'][mask]    # N, 9
    gt_fut_yaw = info['gt_agent_fut_yaw'][mask]      # N, 6
    attr_labels = np.concatenate([gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1).astype(np.float32)

    segmentation, pedestrian = get_birds_eye_view_label(gt_bboxes_3d, attr_labels[None])
    occupancy = np.logical_or(segmentation, pedestrian)

    output_dict = dict(
        gt_bboxes_3d=gt_bboxes_3d,
        gt_names_3d=gt_names_3d,
        gt_attr_labels=attr_labels,
        fut_valid_flag=mask,
        segmentation=segmentation,
        pedestrian=pedestrian,
        occupancy=occupancy
    )

    return output_dict

def vis_traj_on_bev(vis_scenes_infos, vis_path, pred_path, dataset_type, vis_gt=True, vis_bbox=False):
    bev_resolution = 0.4
    center_x, center_y = 100, 100
    ego_width, ego_length = 1.85, 4.084
    W, H = ego_width, ego_length

    for scene_name in vis_scenes_infos:
        scene_infos = vis_scenes_infos[scene_name]
        vis_bev_semantics = []

        save_path = os.path.join(vis_path, scene_name)
        mmcv.mkdir_or_exist(save_path)

        for index, info in enumerate(scene_infos):
            if vis_gt:
                occ_path = info['occ_path']
                if dataset_type == 'openocc':
                    occ_path = occ_path.replace('gts', 'openocc_v2')
                occ_label_path = os.path.join(occ_path, 'labels.npz')
                occ_label = np.load(occ_label_path)
                occ_semantics = occ_label['semantics']
                ego_fut_trajs = info['gt_ego_fut_trajs']
            else:
                token = info['token']
                occ_label_path = os.path.join(pred_path, token+'.npz')
                occ_label = np.load(occ_label_path)
                occ_semantics = occ_label['semantics']
                ego_fut_trajs = occ_label['pred_ego_fut_trajs']

            if vis_bbox:
                bbox_info = get_box_info(info)
                occupancy = bbox_info['occupancy']
                plt.figure()
                for i in range(len(occupancy)):
                    plt.subplot(1, len(occupancy), i+1)
                    plt.axis('off')
                    plt.imshow(occupancy[i])
                plt.show()

            bev_semantics = change_occupancy_to_bev(
                occ_semantics,
                occ_size=(occ_semantics.shape[0], occ_semantics.shape[1], occ_semantics.shape[2]),
                free_cls=16 if dataset_type == 'openocc' else 17,
                colors=openocc_colors_map if dataset_type == 'openocc' else color_map,
                binary=False
            )

            # change ego_fut_trajs to bev_trajs
            ego_fut_trajs_bev = copy.deepcopy(ego_fut_trajs)
            ego_fut_trajs_bev = np.cumsum(ego_fut_trajs_bev, axis=0)
            ego_fut_trajs_bev[:, 0] = ego_fut_trajs_bev[:, 0] / bev_resolution + center_x
            ego_fut_trajs_bev[:, 1] = -ego_fut_trajs_bev[:, 1] / bev_resolution + center_y

            # Utilize spline interpolation to smooth the trajectory
            t = np.linspace(0, 1, len(ego_fut_trajs_bev))
            t_new = np.linspace(0, 1,30)
            spline_x = make_interp_spline(t, ego_fut_trajs_bev[:, 0], k=3)
            spline_y = make_interp_spline(t, ego_fut_trajs_bev[:, 1], k=3)
            x_new = spline_x(t_new)
            y_new = spline_y(t_new)
            smoothed_trajectory = np.column_stack((x_new, y_new)).astype(np.int32)
            cv2.polylines(bev_semantics, [smoothed_trajectory], isClosed=False, color=(128, 128, 128), thickness=1)

            vis_bev_semantics.append(bev_semantics)
            #
            save_path = os.path.join(vis_path, f'{scene_name}/bev_semantics_{str(index).zfill(3)}.png')
            plt.figure()
            plt.imshow(bev_semantics)
            plt.axis('off')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        video_name = 'gt_traj_bev_' + scene_name + '.avi' if vis_gt else 'pred_occ_bev_' + scene_name + '.avi'
        video_path = os.path.join(vis_path, video_name)
        fourcc = cv.VideoWriter_fourcc(*'XVID')

        img_root = os.path.join(vis_path, scene_name)
        img_filelist = os.listdir(img_root)
        demo_img_path = os.path.join(img_root, img_filelist[0])
        demo_img = cv2.imread(demo_img_path)
        video = cv.VideoWriter(video_path, fourcc, 5, (demo_img.shape[0], demo_img.shape[1]))
        for file_path in img_filelist:
            img_path = os.path.join(img_root, file_path)
            img = cv2.imread(img_path)
            video.write(img)
        video.release()

if __name__ == '__main__':
    print('open3d version: {}, if you want to use viewcontrol, make sure using 0.16.0 version!!'.format(o3d.__version__))
    args = parse_args()
    # check vis path
    mmcv.mkdir_or_exist(args.vis_path)

    pkl_data = mmcv.load(args.pkl_file)
    nusc = NuScenes(args.data_version, args.data_path)
    vis_scenes_infos = arange_according_to_scene(pkl_data['infos'], nusc, args.vis_scene)

    # GT Traj visualization
    vis_traj_on_bev(vis_scenes_infos, args.vis_path,  args.pred_path, args.dataset_type, vis_gt=True)
    # GT Traj visualization with bbox
    # vis_traj_on_bev(vis_scenes_infos, args.vis_path,  args.pred_path, args.dataset_type, vis_gt=True, vis_bbox=True)

    # GT visualization
    # vis_scene_occ_on_bev(vis_scenes_infos, args.vis_path,  args.pred_path, args.dataset_type, vis_gt=True)
    # # Prediction visualization
    # vis_scene_occ_on_bev(vis_scenes_infos, args.vis_path,  args.pred_path, args.dataset_type, vis_gt=False)
    # # Visualize single data
    # vis_single_occ_on_bev(args.vis_single_data, args.dataset_type)
