import json

import numpy as np
import torch
import sys
import time
import math
from typing import List, Optional, Sequence, Tuple, Union, Dict

import open3d as o3d
from open3d import geometry
from mmengine.visualization import Visualizer
import matplotlib.pyplot as plt
import colorsys

class LidarVisualizer(Visualizer):
    def __init__(self,
                 name: str = 'visualizer',
                 background_color=(0, 0, 0),
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 color_map=None):
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)
        color_map = np.array(color_map)
        color_map = color_map[:, :3]
        self.background_color = background_color
        self.color_map = color_map

        self.flag_pause = False
        self.flag_next = False
        self.flag_exit = False

    def escape_callback(self, vis):
        self.o3d_vis.clear_geometries()
        self.o3d_vis.destroy_window()
        self.o3d_vis.close()
        self._clear_o3d_vis()
        sys.exit(0)

    def right_callback(self, vis):
        self.flag_next = True
        return False

    def _clear_o3d_vis(self) -> None:
        """Clear open3d vis."""

        if hasattr(self, 'o3d_vis'):
            del self.o3d_vis
            del self.points_colors
            del self.view_control
            if hasattr(self, 'pcd'):
                del self.pcd

    def space_action_callback(self, vis, action, mods):
        if action == 1:
            if self.flag_pause:
                print(
                    'Playback continued, press [SPACE] to pause.', )
            else:
                print(
                    'Playback paused, press [SPACE] to continue.')
            self.flag_pause = not self.flag_pause
        return True

    def _initialize_o3d_vis(self):
        """Initialize open3d vis according to frame_cfg.

        Args:
            frame_cfg (dict): The config to create coordinate frame in open3d
                vis.

        Returns:
            :obj:`o3d.visualization.Visualizer`: Created open3d vis.
        """
        if o3d is None or geometry is None:
            raise ImportError(
                'Please run "pip install open3d" to install open3d first.')
        glfw_key_escape = 256  # Esc
        glfw_key_space = 32  # Space
        glfw_key_right = 262  # Right
        o3d_vis = o3d.visualization.VisualizerWithKeyCallback()
        o3d_vis.register_key_callback(glfw_key_escape, self.escape_callback)
        o3d_vis.register_key_action_callback(glfw_key_space,
                                             self.space_action_callback)
        o3d_vis.register_key_callback(glfw_key_right, self.right_callback)
        o3d_vis.create_window(width=1920, height=1080)

        self.view_control = o3d_vis.get_view_control()

        return o3d_vis

    def vis_lidar_points(self,
                         lidar_points,
                         points_size=2,
                         mode='xyz',
                         points_color: Tuple[float] = (0.8, 0.8, 0.8),
                         frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
                         save_path=None,
                         wait_time=-1,
                         view_json=None
                         ):
        if not hasattr(self, 'o3d_vis'):
            self.o3d_vis = self._initialize_o3d_vis()

        # set points size in Open3D
        render_option = self.o3d_vis.get_render_option()
        if render_option is not None:
            render_option.point_size = points_size
            render_option.background_color = np.asarray([0, 0, 0])

        points = lidar_points.copy()
        pcd = geometry.PointCloud()
        if mode == 'xyz':
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            points_colors = np.tile(
                np.array(points_color), (points.shape[0], 1))
        elif mode == 'xyzrgb':
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            points_colors = points[:, 3:6]
            # normalize to [0, 1] for Open3D drawing
            if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
                points_colors /= 255.0
        else:
            raise NotImplementedError

        # create coordinate frame
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(**frame_cfg)
        self.o3d_vis.add_geometry(mesh_frame)

        self.o3d_vis.add_geometry(mesh_frame)

        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        self.o3d_vis.add_geometry(pcd)
        self.pcd = pcd
        self.points_colors = points_colors

        self.show(save_path=save_path, wait_time=wait_time, view_json=view_json)


    def show(self,
             save_path: Optional[str] = None,
             wait_time: int = -1,
             view_json=None) -> None:

        if view_json is not None:
            self.view_control.convert_from_pinhole_camera_parameters(view_json)

        if hasattr(self, 'o3d_vis'):
            if hasattr(self, 'view_port'):
                self.view_control.convert_from_pinhole_camera_parameters(
                    self.view_port)
            self.flag_exit = not self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer()
            self.view_port = \
                self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
            if wait_time != -1:
                self.last_time = time.time()
                while time.time(
                ) - self.last_time < wait_time and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
                while self.flag_pause and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501

            else:
                while not self.flag_next and self.o3d_vis.poll_events():
                    self.o3d_vis.update_renderer()
                    self.view_port = \
                        self.view_control.convert_to_pinhole_camera_parameters()  # noqa: E501
                self.flag_next = False
            self.o3d_vis.clear_geometries()
            try:
                del self.pcd
            except (KeyError, AttributeError):
                pass
            if save_path is not None:
                if not (save_path.endswith('.png')
                        or save_path.endswith('.jpg')):
                    save_path += '.png'
                self.o3d_vis.capture_screen_image(save_path)

            param = self.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters('view.json', param)

            if self.flag_exit:
                self.o3d_vis.destroy_window()
                self.o3d_vis.close()
                self._clear_o3d_vis()
                sys.exit(0)
