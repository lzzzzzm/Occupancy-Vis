"""
3D Occupancy Grid Visualizer Module

A professional visualization toolkit for 3D occupancy grids with semantic segmentation
and optical flow support using Open3D.

Author: Zhimin-Liao
Version: 2.0
License: MIT
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import colorsys
import numpy as np

# Optional imports with graceful fallback
try:
    import open3d as o3d
    from open3d import geometry
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None
    geometry = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationMode(Enum):
    """Enumeration for different visualization modes."""
    SEMANTIC = "semantic"
    FLOW = "flow"
    COMBINED = "combined"


class RenderQuality(Enum):
    """Enumeration for rendering quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class VisualizationConfig:
    """Configuration class for visualization parameters."""
    voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)
    range_vals: Tuple[float, ...] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
    ignore_labels: List[int] = None
    background_color: Tuple[int, int, int] = (255, 255, 255)
    point_size: int = 4
    show_ego_car: bool = True
    show_coordinate_frame: bool = True
    frame_size: float = 1.0
    wait_time: float = -1
    auto_save: bool = False

    def __post_init__(self):
        if self.ignore_labels is None:
            self.ignore_labels = [0, 17]


@dataclass
class CameraParameters:
    """Camera parameters for visualization."""
    width: int = 1920
    height: int = 1080
    view_json_path: Optional[Union[str, Path]] = None


class OccupancyVisualizerError(Exception):
    """Custom exception for occupancy visualizer errors."""
    pass


class DependencyError(OccupancyVisualizerError):
    """Exception raised when required dependencies are missing."""
    pass


def require_open3d(func):
    """Decorator to ensure Open3D is available before calling a function."""
    def wrapper(*args, **kwargs):
        if not OPEN3D_AVAILABLE:
            raise DependencyError(
                "Open3D is required for this functionality. "
                "Please install it with: pip install open3d"
            )
        return func(*args, **kwargs)
    return wrapper


def require_cv2(func):
    """Decorator to ensure OpenCV is available before calling a function."""
    def wrapper(*args, **kwargs):
        if not CV2_AVAILABLE:
            raise DependencyError(
                "OpenCV is required for this functionality. "
                "Please install it with: pip install opencv-python"
            )
        return func(*args, **kwargs)
    return wrapper


class BaseComponent(ABC):
    """Abstract base class for all visualizer components."""

    @abstractmethod
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate input parameters."""
        pass


class CameraController(BaseComponent):
    """
    Camera controller for managing view parameters and camera operations.

    This class handles all camera-related operations including loading/saving
    camera parameters, view control, and camera state management.
    """

    def __init__(self, config: CameraParameters):
        """
        Initialize camera controller.

        Args:
            config: Camera configuration parameters
        """
        self.config = config
        self._view_control = None
        self._current_parameters = None

    def validate_inputs(self, view_json: Union[str, Path]) -> bool:
        """Validate camera parameter file path."""
        if isinstance(view_json, (str, Path)):
            path = Path(view_json)
            return path.exists() and path.suffix == '.json'
        return False

    @require_open3d
    def load_camera_parameters(self, view_json: Union[str, Path]) -> o3d.camera.PinholeCameraParameters:
        """
        Load camera parameters from JSON file.

        Args:
            view_json: Path to the JSON file containing camera parameters

        Returns:
            Open3D pinhole camera parameters

        Raises:
            OccupancyVisualizerError: If loading fails
        """
        if not self.validate_inputs(view_json):
            raise OccupancyVisualizerError(f"Invalid camera parameter file: {view_json}")

        try:
            params = o3d.io.read_pinhole_camera_parameters(str(view_json))
            self._current_parameters = params
            logger.info(f"Successfully loaded camera parameters from {view_json}")
            return params
        except Exception as e:
            raise OccupancyVisualizerError(f"Failed to load camera parameters from {view_json}: {e}")

    @require_open3d
    def save_camera_parameters(self,
                             view_control: o3d.visualization.ViewControl,
                             save_path: Union[str, Path] = 'view.json') -> None:
        """
        Save current camera parameters to JSON file.

        Args:
            view_control: Open3D view control object
            save_path: Path where to save the parameters

        Raises:
            OccupancyVisualizerError: If saving fails
        """
        try:
            param = view_control.convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters(str(save_path), param)
            self._current_parameters = param
            logger.info(f"Camera parameters saved to {save_path}")
        except Exception as e:
            raise OccupancyVisualizerError(f"Failed to save camera parameters: {e}")

    def apply_parameters(self, view_control: o3d.visualization.ViewControl) -> None:
        """Apply stored camera parameters to view control."""
        if self._current_parameters is not None:
            try:
                view_control.convert_from_pinhole_camera_parameters(self._current_parameters)
            except Exception as e:
                logger.warning(f"Failed to apply camera parameters: {e}")

    def load_external_parameters(self, view_json_path: Union[str, Path]) -> None:
        """
        Load and store camera parameters from external file for reuse.

        This method loads camera parameters from an external view.json file
        and stores them for consistent application across multiple visualizations.

        Args:
            view_json_path: Path to the external view.json file

        Example:
            camera_controller.load_external_parameters('my_saved_view.json')
            # Now all subsequent visualizations will use these parameters
        """
        params = self.load_camera_parameters(view_json_path)
        self._current_parameters = params
        logger.info(f"External camera parameters loaded and will be applied to all visualizations")

    def has_parameters(self) -> bool:
        """Check if camera parameters are currently loaded."""
        return self._current_parameters is not None

    def clear_parameters(self) -> None:
        """Clear currently stored camera parameters."""
        self._current_parameters = None
        logger.info("Camera parameters cleared")

    def get_parameters_info(self) -> Optional[Dict]:
        """
        Get information about currently loaded camera parameters.

        Returns:
            Dictionary with camera parameter information or None if no parameters loaded
        """
        if self._current_parameters is None:
            return None

        try:
            intrinsic = self._current_parameters.intrinsic
            extrinsic = self._current_parameters.extrinsic

            return {
                "intrinsic_matrix": np.array(intrinsic.intrinsic_matrix).tolist(),
                "width": intrinsic.width,
                "height": intrinsic.height,
                "extrinsic_matrix": extrinsic.tolist(),
                "has_parameters": True
            }
        except Exception as e:
            logger.warning(f"Could not extract parameter info: {e}")
            return {"has_parameters": True, "details": "Available but not readable"}


class GeometryGenerator(BaseComponent):
    """
    Geometry generator for creating 3D geometric objects.

    This class handles the generation of various 3D geometries including
    ego car models, bounding boxes, and coordinate frames.
    """

    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate geometry generation inputs."""
        return True  # Basic validation, can be extended

    @staticmethod
    def generate_ego_car(ego_range: Optional[List[float]] = None,
                        ego_voxel_size: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate ego car model as point cloud.

        Args:
            ego_range: Bounding box of ego car [x_min, y_min, z_min, x_max, y_max, z_max]
            ego_voxel_size: Voxel size for ego car discretization [dx, dy, dz]

        Returns:
            Point cloud array with shape (N, 6) containing [x, y, z, r, g, b]
        """
        # Set default values
        if ego_range is None:
            ego_range = [-1, -1, 0, 3, 1, 1.5]
        if ego_voxel_size is None:
            ego_voxel_size = [0.1, 0.1, 0.1]

        # Calculate dimensions
        ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
        ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
        ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])

        # Generate grid coordinates
        temp_x = np.arange(ego_xdim)
        temp_y = np.arange(ego_ydim)
        temp_z = np.arange(ego_zdim)
        ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)

        # Convert to world coordinates
        ego_point_x = ((ego_xyz[:, 0:1] + 0.5) / ego_xdim *
                      (ego_range[3] - ego_range[0]) + ego_range[0])
        ego_point_y = ((ego_xyz[:, 1:2] + 0.5) / ego_ydim *
                      (ego_range[4] - ego_range[1]) + ego_range[1])
        ego_point_z = ((ego_xyz[:, 2:3] + 0.5) / ego_zdim *
                      (ego_range[5] - ego_range[2]) + ego_range[2])

        ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)

        # Generate rainbow colors based on height
        normalized_z = (ego_point_z - ego_range[2]) / (ego_range[5] - ego_range[2])
        ego_point_rgb = np.concatenate((
            normalized_z,
            np.zeros_like(normalized_z),
            1 - normalized_z
        ), axis=-1) * 255

        return np.concatenate((ego_point_xyz, ego_point_rgb), axis=-1)

    @staticmethod
    def compute_box_3d(center: np.ndarray,
                      size: np.ndarray,
                      heading_angle: np.ndarray) -> np.ndarray:
        """
        Compute 3D bounding box corners from center, size, and heading.

        Args:
            center: Box centers with shape (N, 3)
            size: Box sizes with shape (N, 3) [width, length, height]
            heading_angle: Heading angles with shape (N, 1)

        Returns:
            Box corners with shape (N, 8, 3)
        """
        if center.ndim == 1:
            center = center.reshape(1, -1)
        if size.ndim == 1:
            size = size.reshape(1, -1)
        if heading_angle.ndim == 1:
            heading_angle = heading_angle.reshape(-1, 1)

        h, w, l = size[:, 2], size[:, 0], size[:, 1]
        heading_angle = -heading_angle - math.pi / 2

        # Adjust center to bottom center
        center_adjusted = center.copy()
        center_adjusted[:, 2] = center_adjusted[:, 2] + h / 2

        # Half dimensions
        l_half = (l / 2).reshape(-1, 1)
        w_half = (w / 2).reshape(-1, 1)
        h_half = (h / 2).reshape(-1, 1)

        # Define corners in local coordinate system
        x_corners = np.concatenate([-l_half, l_half, l_half, -l_half,
                                   -l_half, l_half, l_half, -l_half], axis=1)[..., None]
        y_corners = np.concatenate([w_half, w_half, -w_half, -w_half,
                                   w_half, w_half, -w_half, -w_half], axis=1)[..., None]
        z_corners = np.concatenate([h_half, h_half, h_half, h_half,
                                   -h_half, -h_half, -h_half, -h_half], axis=1)[..., None]

        # Combine corners
        corners_3d = np.concatenate([x_corners, y_corners, z_corners], axis=2)

        # Apply rotation (if needed) and translation
        corners_3d[..., 0] += center_adjusted[:, 0:1]
        corners_3d[..., 1] += center_adjusted[:, 1:2]
        corners_3d[..., 2] += center_adjusted[:, 2:3]

        return corners_3d

    @require_open3d
    def create_coordinate_frame(self, size: float = 1.0,
                               origin: List[float] = None) -> o3d.geometry.TriangleMesh:
        """
        Create coordinate frame mesh.

        Args:
            size: Size of the coordinate frame
            origin: Origin position [x, y, z]

        Returns:
            Open3D triangle mesh representing coordinate frame
        """
        if origin is None:
            origin = [0, 0, 0]

        frame = geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=origin
        )
        return frame


class MeshLoader(BaseComponent):
    """
    Mesh loader for handling 3D model files.

    This class manages loading and preprocessing of 3D mesh models,
    particularly for car models and other geometric objects.
    """

    def validate_inputs(self, mesh_path: Union[str, Path]) -> bool:
        """Validate mesh file path."""
        if mesh_path is None:
            return True  # None is acceptable
        path = Path(mesh_path)
        return path.exists() and path.suffix.lower() in ['.obj', '.ply', '.stl']

    @require_open3d
    def load_mesh(self, mesh_path: Union[str, Path]) -> Optional[o3d.geometry.TriangleMesh]:
        """
        Load and preprocess 3D mesh model.

        Args:
            mesh_path: Path to the mesh file

        Returns:
            Loaded and processed triangle mesh, or None if loading fails

        Raises:
            OccupancyVisualizerError: If mesh loading fails
        """
        if mesh_path is None:
            return None

        if not self.validate_inputs(mesh_path):
            raise OccupancyVisualizerError(f"Invalid mesh file: {mesh_path}")

        try:
            # Load mesh
            car_model_mesh = o3d.io.read_triangle_mesh(str(mesh_path))

            if len(car_model_mesh.vertices) == 0:
                logger.warning(f"Loaded mesh from {mesh_path} is empty")
                return None

            # Apply transformations
            self._preprocess_car_mesh(car_model_mesh)

            logger.info(f"Successfully loaded mesh from {mesh_path}")
            return car_model_mesh

        except Exception as e:
            raise OccupancyVisualizerError(f"Failed to load mesh from {mesh_path}: {e}")

    def _preprocess_car_mesh(self, mesh: o3d.geometry.TriangleMesh) -> None:
        """Apply standard preprocessing to car mesh."""
        # Rotate 90 degrees around X-axis
        angle = np.pi / 2
        R = mesh.get_rotation_matrix_from_axis_angle(np.array([angle, 0, 0]))
        mesh.rotate(R, center=mesh.get_center())

        # Scale down
        mesh.scale(0.25, center=mesh.get_center())

        # Move to standard position
        current_center = mesh.get_center()
        new_center = np.array([0, 0, 0.5])
        translation = new_center - current_center
        mesh.translate(translation)

        # Compute normals for proper lighting
        mesh.compute_vertex_normals()


class ColorProcessor(BaseComponent):
    """
    Color processing utilities for visualization.

    This class handles color mapping, flow visualization, and color space
    conversions for different visualization modes.
    """

    def validate_inputs(self, vx: np.ndarray, vy: np.ndarray) -> bool:
        """Validate flow vector inputs."""
        return (isinstance(vx, np.ndarray) and isinstance(vy, np.ndarray) and
                vx.shape == vy.shape)

    @staticmethod
    def flow_to_color(vx: np.ndarray,
                     vy: np.ndarray,
                     max_magnitude: Optional[float] = None) -> np.ndarray:
        """
        Convert optical flow vectors to color representation.

        Args:
            vx: Flow vectors in x direction
            vy: Flow vectors in y direction
            max_magnitude: Maximum magnitude for normalization

        Returns:
            RGB color array with shape (*vx.shape, 3)
        """
        # Calculate magnitude and angle
        magnitude = np.sqrt(vx ** 2 + vy ** 2)
        angle = np.arctan2(vy, vx)

        # Map angle to hue (0-1)
        hue = (angle + np.pi) / (2 * np.pi)

        # Map magnitude to saturation
        if max_magnitude is None:
            max_magnitude = np.max(magnitude) if magnitude.size > 0 else 1.0

        if max_magnitude == 0:
            saturation = np.zeros_like(magnitude)
        else:
            saturation = np.clip(magnitude / max_magnitude + 1e-3, 0, 1)

        value = np.ones_like(saturation)

        # Convert HSV to RGB
        hsv = np.stack((hue, saturation, value), axis=-1)
        rgb = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), -1, hsv)

        # Handle NaN values
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)

        return (rgb * 255).astype(np.uint8)

    def create_legend_circle(self,
                           radius: float = 1.0,
                           resolution: int = 500) -> np.ndarray:
        """
        Create circular legend for flow visualization.

        Args:
            radius: Radius of the legend circle
            resolution: Resolution of the output image

        Returns:
            Legend image array with shape (resolution, resolution, 3)
        """
        # Create coordinate grid
        x = np.linspace(-radius, radius, resolution)
        y = np.linspace(-radius, radius, resolution)
        X, Y = np.meshgrid(x, y)

        # Calculate flow vectors
        vx, vy = X, Y
        magnitude = np.sqrt(vx ** 2 + vy ** 2)
        mask = magnitude <= radius

        # Get colors for valid points
        vx_valid = vx[mask]
        vy_valid = vy[mask]
        colors = self.flow_to_color(vx_valid, vy_valid, max_magnitude=radius)

        # Create legend image
        legend_image = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
        mask_2d = mask.reshape(resolution, resolution)
        legend_image[mask_2d] = colors

        return legend_image


class VoxelProcessor(BaseComponent):
    """
    Voxel processing utilities for occupancy data.

    This class handles conversion between voxel representations and point clouds,
    voxel filtering, and geometric transformations.
    """

    def validate_inputs(self, voxel: np.ndarray) -> bool:
        """Validate voxel input array."""
        return isinstance(voxel, np.ndarray) and voxel.ndim >= 3

    @staticmethod
    def voxel_to_points(voxel: np.ndarray,
                       voxel_flow: Optional[np.ndarray] = None,
                       voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
                       range_vals: Tuple[float, ...] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
                       ignore_labels: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Convert voxel occupancy data to point cloud.

        Args:
            voxel: Occupancy voxel array with shape (H, W, D)
            voxel_flow: Optional flow data with same spatial shape as voxel
            voxel_size: Size of each voxel [dx, dy, dz]
            range_vals: Spatial range [x_min, y_min, z_min, x_max, y_max, z_max]
            ignore_labels: List of labels to filter out

        Returns:
            Tuple of (points, labels, flow_vectors)
            - points: Point coordinates with shape (N, 3)
            - labels: Voxel labels with shape (N,)
            - flow_vectors: Flow vectors with shape (N, 2) or None
        """
        if ignore_labels is None:
            ignore_labels = [17, 255]

        # Ensure numpy array
        if not isinstance(voxel, np.ndarray):
            voxel = np.array(voxel)

        # Create mask for valid voxels
        mask = np.ones_like(voxel, dtype=bool)
        for ignore_label in ignore_labels:
            mask &= (voxel != ignore_label)

        # Get occupied voxel indices
        occ_idx = np.where(mask)

        if len(occ_idx[0]) == 0:
            # Return empty arrays if no valid voxels
            empty_points = np.empty((0, 3))
            empty_labels = np.empty((0,))
            empty_flow = np.empty((0, 2)) if voxel_flow is not None else None
            return empty_points, empty_labels, empty_flow

        # Convert indices to world coordinates
        points = np.column_stack((
            occ_idx[0] * voxel_size[0] + voxel_size[0] / 2 + range_vals[0],
            occ_idx[1] * voxel_size[1] + voxel_size[1] / 2 + range_vals[1],
            occ_idx[2] * voxel_size[2] + voxel_size[2] / 2 + range_vals[2]
        ))

        # Extract labels and flow
        labels = voxel[occ_idx]
        flow = voxel_flow[occ_idx] if voxel_flow is not None else None

        return points, labels, flow

    @staticmethod
    def get_voxel_profile(points: np.ndarray,
                         voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4)) -> np.ndarray:
        """
        Generate voxel profile for bounding box visualization.

        Args:
            points: Point coordinates with shape (N, 3)
            voxel_size: Size of each voxel [dx, dy, dz]

        Returns:
            Voxel profiles with shape (N, 7) [x, y, z, w, l, h, yaw]
        """
        if points.shape[0] == 0:
            return np.empty((0, 7))

        # Centers (adjust z to bottom of voxel)
        centers = np.column_stack((
            points[:, :2],
            points[:, 2] - voxel_size[2] / 2
        ))

        # Dimensions (width, length, height)
        wlh = np.column_stack((
            np.full(centers.shape[0], voxel_size[0]),
            np.full(centers.shape[0], voxel_size[1]),
            np.full(centers.shape[0], voxel_size[2])
        ))

        # No rotation for voxels
        yaw = np.zeros((centers.shape[0], 1))

        return np.column_stack((centers, wlh, yaw))


class InputValidator:
    """Input validation utilities for the visualizer."""

    @staticmethod
    def validate_voxel_data(voxel: np.ndarray) -> None:
        """Validate voxel input data."""
        if not isinstance(voxel, np.ndarray):
            raise OccupancyVisualizerError("Voxel data must be a numpy array")
        if voxel.ndim != 3:
            raise OccupancyVisualizerError(f"Voxel data must be 3D, got {voxel.ndim}D")
        if voxel.size == 0:
            raise OccupancyVisualizerError("Voxel data cannot be empty")

    @staticmethod
    def validate_flow_data(flow: np.ndarray, voxel_shape: Tuple[int, ...]) -> None:
        """Validate flow input data."""
        if not isinstance(flow, np.ndarray):
            raise OccupancyVisualizerError("Flow data must be a numpy array")
        expected_shape = voxel_shape + (2,)
        if flow.shape != expected_shape:
            raise OccupancyVisualizerError(
                f"Flow data shape {flow.shape} doesn't match expected {expected_shape}"
            )

    @staticmethod
    def validate_color_map(color_map: Optional[np.ndarray], num_classes: int) -> None:
        """Validate color map."""
        if color_map is not None:
            if not isinstance(color_map, np.ndarray):
                raise OccupancyVisualizerError("Color map must be a numpy array")
            if color_map.ndim != 2 or color_map.shape[1] != 3:
                raise OccupancyVisualizerError("Color map must have shape (N, 3)")
            if color_map.shape[0] < num_classes:
                logger.warning(f"Color map has {color_map.shape[0]} colors but {num_classes} classes")


class BEVProcessor(BaseComponent):
    """
    Bird's Eye View (BEV) processing for occupancy data.

    This class converts 3D occupancy grids into 2D top-down views by finding
    the topmost non-free voxel along the height dimension for each (x, y) location.
    Supports different datasets (OpenOCC, NuScenes) with configurable free classes.
    """

    def __init__(self, color_map: Optional[np.ndarray] = None):
        """
        Initialize BEV processor.

        Args:
            color_map: Optional color map for BEV visualization. If None, will use default colors.
        """
        super().__init__()
        self.color_map = color_map

    def validate_inputs(self, occupancy_data: np.ndarray) -> bool:
        """Validate occupancy data for BEV conversion."""
        if not isinstance(occupancy_data, np.ndarray):
            return False
        if occupancy_data.ndim != 3:
            return False
        return True

    def convert_to_bev(self,
                      occupancy_data: np.ndarray,
                      free_class: int = 17,
                      color_map: Optional[np.ndarray] = None,
                      dataset_type: str = 'nuscenes') -> np.ndarray:
        """
        Convert 3D occupancy data to Bird's Eye View (BEV) format.

        This method projects 3D occupancy data to a 2D top-down view by finding
        the topmost non-free voxel for each (x, y) location. The algorithm:
        1. Create a depth map prioritizing higher voxels
        2. Find the topmost non-free voxel using argmax
        3. Gather semantic labels from the selected voxels
        4. Apply color mapping for visualization

        Args:
            occupancy_data: 3D occupancy grid with shape (H, W, D)
            free_class: Label value representing free/empty space
            color_map: Color map for visualization (BGR format for OpenCV)
            dataset_type: Dataset type for specific optimizations ('nuscenes', 'openocc')

        Returns:
            BEV image as numpy array with shape (H, W, 3) in BGR format

        Raises:
            OccupancyVisualizerError: If input validation fails
        """
        if not self.validate_inputs(occupancy_data):
            raise OccupancyVisualizerError("Invalid occupancy data for BEV conversion")

        try:
            # Use provided color map or instance color map
            colors = color_map if color_map is not None else self.color_map
            if colors is None:
                raise OccupancyVisualizerError("No color map provided for BEV visualization")

            H, W, D = occupancy_data.shape
            logger.debug(f"Converting occupancy data to BEV: shape={occupancy_data.shape}, free_class={free_class}")

            # Create mask for non-free voxels
            semantics_valid = np.logical_not(occupancy_data == free_class)

            # Create depth priority map (higher voxels get higher priority)
            depth_map = np.arange(D, dtype=np.float32).reshape(1, 1, D)
            depth_map = np.broadcast_to(depth_map, (H, W, D))

            # Apply validity mask (free voxels get 0 priority)
            depth_map = depth_map * semantics_valid.astype(np.float32)

            # Find topmost non-free voxel for each (x, y) location
            selected_indices = np.argmax(depth_map, axis=2)

            # Gather semantic labels from selected voxels
            # Create indices for advanced indexing
            h_indices = np.arange(H)[:, np.newaxis]
            w_indices = np.arange(W)[np.newaxis, :]

            # Extract semantic labels at selected heights
            bev_semantics = occupancy_data[h_indices, w_indices, selected_indices]

            # Handle edge case: if all voxels in a column are free, use free_class
            all_free_mask = np.all(occupancy_data == free_class, axis=2)
            bev_semantics[all_free_mask] = free_class

            # Apply color mapping
            bev_semantics_flat = bev_semantics.flatten().astype(np.int32)

            # Ensure indices are within color map bounds
            bev_semantics_flat = np.clip(bev_semantics_flat, 0, len(colors) - 1)

            # Apply colors
            bev_colored = colors[bev_semantics_flat].astype(np.uint8)
            bev_image = bev_colored.reshape(H, W, 3)

            # Apply dataset-specific transformations
            bev_image = self._apply_dataset_transforms(bev_image, dataset_type)

            logger.debug(f"BEV conversion completed: output_shape={bev_image.shape}")
            return bev_image

        except Exception as e:
            raise OccupancyVisualizerError(f"BEV conversion failed: {e}")

    def _apply_dataset_transforms(self, bev_image: np.ndarray, dataset_type: str) -> np.ndarray:
        """Apply dataset-specific transformations to BEV image."""
        try:
            # Apply transformations based on original implementation
            # Flip both axes to match expected orientation
            transformed = bev_image[::-1, ::-1, :3]

            # Dataset-specific adjustments
            if dataset_type.lower() == 'openocc':
                # OpenOCC-specific transformations if needed
                pass
            elif dataset_type.lower() == 'nuscenes':
                # NuScenes-specific transformations if needed
                pass

            return transformed

        except Exception as e:
            logger.warning(f"Dataset transform failed: {e}, using original image")
            return bev_image

    def create_bev_with_metadata(self,
                                occupancy_data: np.ndarray,
                                free_class: int = 17,
                                color_map: Optional[np.ndarray] = None,
                                dataset_type: str = 'nuscenes',
                                add_text_overlay: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Create BEV visualization with metadata overlay.

        Args:
            occupancy_data: 3D occupancy grid
            free_class: Free space class label
            color_map: Color map for visualization
            dataset_type: Dataset type
            add_text_overlay: Whether to add informational text overlay

        Returns:
            Tuple of (bev_image, metadata_dict)
        """
        # Create basic BEV
        bev_image = self.convert_to_bev(occupancy_data, free_class, color_map, dataset_type)

        # Collect metadata
        metadata = {
            'original_shape': occupancy_data.shape,
            'bev_shape': bev_image.shape,
            'free_class': free_class,
            'dataset_type': dataset_type,
            'unique_classes': np.unique(occupancy_data).tolist(),
            'occupancy_ratio': np.mean(occupancy_data != free_class)
        }

        # Add text overlay if requested
        if add_text_overlay:
            bev_image = self._add_text_overlay(bev_image, metadata)

        return bev_image, metadata

    @require_cv2
    def _add_text_overlay(self, bev_image: np.ndarray, metadata: Dict) -> np.ndarray:
        """Add informational text overlay to BEV image."""
        try:
            image_with_text = bev_image.copy()

            # Text configuration
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (255, 255, 255)  # White text
            thickness = 1

            # Add title
            title = f"BEV - {metadata['dataset_type'].upper()}"
            cv2.putText(image_with_text, title, (10, 25), font, font_scale, color, thickness)

            # Add shape info
            shape_text = f"Shape: {metadata['original_shape']} -> {metadata['bev_shape'][:2]}"
            cv2.putText(image_with_text, shape_text, (10, 50), font, 0.5, color, thickness)

            # Add occupancy ratio
            occ_text = f"Occupancy: {metadata['occupancy_ratio']:.2%}"
            cv2.putText(image_with_text, occ_text, (10, 70), font, 0.5, color, thickness)

            return image_with_text

        except Exception as e:
            logger.warning(f"Failed to add text overlay: {e}")
            return bev_image

    def process_bev_sequence(self,
                           occupancy_sequence: List[np.ndarray],
                           free_class: int = 17,
                           color_map: Optional[np.ndarray] = None,
                           dataset_type: str = 'nuscenes') -> List[np.ndarray]:
        """
        Process a sequence of occupancy data into BEV format.

        Args:
            occupancy_sequence: List of 3D occupancy grids
            free_class: Free space class label
            color_map: Color map for visualization
            dataset_type: Dataset type

        Returns:
            List of BEV images
        """
        bev_sequence = []

        for i, occupancy_data in enumerate(occupancy_sequence):
            try:
                bev_image = self.convert_to_bev(
                    occupancy_data, free_class, color_map, dataset_type
                )
                bev_sequence.append(bev_image)
                logger.debug(f"Processed BEV frame {i + 1}/{len(occupancy_sequence)}")

            except Exception as e:
                logger.error(f"Failed to process BEV frame {i}: {e}")
                continue

        logger.info(f"BEV sequence processing completed: {len(bev_sequence)}/{len(occupancy_sequence)} frames")
        return bev_sequence

    @staticmethod
    def get_dataset_free_class(dataset_type: str) -> int:
        """Get the default free class for different datasets."""
        dataset_free_classes = {
            'openocc': 16,
            'nuscenes': 17,
            'occ3d': 17,
            'default': 17
        }
        return dataset_free_classes.get(dataset_type.lower(), 17)

    @staticmethod
    def create_for_dataset(dataset_type: str, color_map: Optional[np.ndarray] = None) -> 'BEVProcessor':
        """
        Create BEV processor optimized for specific dataset.

        Args:
            dataset_type: Dataset type ('openocc', 'nuscenes', 'occ3d')
            color_map: Optional color map override

        Returns:
            Configured BEV processor
        """
        if color_map is None:
            # Use dataset-appropriate color map
            if dataset_type.lower() == 'openocc':
                from utils_color import openocc_colors_map
                color_map = openocc_colors_map
            else:
                color_map = get_occ3d_color_map()  # Default to OCC3D colors

        processor = BEVProcessor(color_map=color_map)
        logger.info(f"Created BEV processor for {dataset_type} dataset")
        return processor


class EventHandler:
    """Event handling for interactive visualization."""

    def __init__(self, visualizer: 'OccupancyVisualizer'):
        self.visualizer = visualizer
        self.setup_key_bindings()

    def setup_key_bindings(self) -> Dict[int, callable]:
        """Setup keyboard event handlers."""
        # GLFW key codes
        self.key_bindings = {
            256: self._on_escape,      # ESC
            32: self._on_space,        # SPACE
            262: self._on_right_arrow, # RIGHT
            263: self._on_left_arrow,  # LEFT
            83: self._on_s_key,        # S (save)
            82: self._on_r_key,        # R (reset view)
        }
        return self.key_bindings

    def _on_escape(self, vis) -> bool:
        """Handle escape key press."""
        self.visualizer.flag_exit = True
        return False

    def _on_space(self, vis) -> bool:
        """Handle space key press (pause/resume)."""
        self.visualizer.flag_pause = not self.visualizer.flag_pause
        status = "paused" if self.visualizer.flag_pause else "resumed"
        logger.info(f"Playback {status}")
        return True

    def _on_right_arrow(self, vis) -> bool:
        """Handle right arrow key press (next frame)."""
        self.visualizer.flag_next = True
        return False

    def _on_left_arrow(self, vis) -> bool:
        """Handle left arrow key press (previous frame)."""
        # Implement if needed
        return True

    def _on_s_key(self, vis) -> bool:
        """Handle S key press (save screenshot)."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = f"screenshot_{timestamp}.png"
            vis.capture_screen_image(save_path)
            logger.info(f"Screenshot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
        return True

    def _on_r_key(self, vis) -> bool:
        """Handle R key press (reset view)."""
        try:
            vis.get_view_control().reset_view_point(True)
            logger.info("View reset to default")
        except Exception as e:
            logger.error(f"Failed to reset view: {e}")
        return True


class RenderManager:
    """Manages rendering operations and scene setup."""

    # Predefined bounding box edges for wireframe rendering
    BOX_EDGES = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Bottom face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ])

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.geometry_generator = GeometryGenerator()
        self.mesh_loader = MeshLoader()

    @require_open3d
    def create_point_cloud(self,
                          points: np.ndarray,
                          colors: np.ndarray,
                          ego_points: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """
        Create Open3D point cloud from points and colors.

        Args:
            points: Point coordinates with shape (N, 3)
            colors: Point colors with shape (N, 3)
            ego_points: Optional ego car points with shape (M, 6) [x,y,z,r,g,b]

        Returns:
            Open3D point cloud object
        """
        # Combine with ego points if provided
        if ego_points is not None:
            points = np.concatenate([points, ego_points[:, :3]], axis=0)
            ego_colors = ego_points[:, 3:6]
            colors = np.concatenate([colors, ego_colors], axis=0)

        # Create point cloud
        pcd = geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Handle color normalization
        if colors.dtype != np.float64:
            colors = colors.astype(np.float64)

        if colors.max() > 1.0:
            colors = colors / 255.0

        # Convert RGB to BGR for Open3D
        colors = colors[:, [2, 1, 0]]  # RGB -> BGR
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    @require_open3d
    def create_voxel_grid(self,
                         pcd: o3d.geometry.PointCloud,
                         voxel_size: float = 0.4) -> o3d.geometry.VoxelGrid:
        """Create voxel grid from point cloud."""
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=voxel_size
        )
        return voxel_grid

    @require_open3d
    def create_bounding_boxes(self, bbox_corners: np.ndarray) -> o3d.geometry.LineSet:
        """
        Create bounding box wireframes.

        Args:
            bbox_corners: Bounding box corners with shape (N, 8, 3)

        Returns:
            Open3D line set representing bounding boxes
        """
        if bbox_corners.shape[0] == 0:
            # Return empty line set
            return o3d.geometry.LineSet()

        num_boxes = bbox_corners.shape[0]

        # Create edge indices for all boxes
        bases = np.arange(0, num_boxes * 8, 8)
        edges = np.tile(self.BOX_EDGES.reshape((1, 12, 2)), (num_boxes, 1, 1))
        edges = edges + bases[:, None, None]

        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3)))
        line_set.lines = o3d.utility.Vector2iVector(edges.reshape((-1, 2)))
        line_set.paint_uniform_color((0, 0, 0))  # Black wireframes

        return line_set

    def setup_render_options(self,
                            vis: o3d.visualization.Visualizer,
                            quality: RenderQuality = RenderQuality.HIGH) -> None:
        """Setup rendering options based on quality level."""
        render_option = vis.get_render_option()
        if render_option is None:
            return

        # Set background color
        render_option.background_color = np.asarray(self.config.background_color) / 255.0

        # Quality-based settings
        quality_settings = {
            RenderQuality.LOW: {
                'point_size': max(1, self.config.point_size // 2),
                'line_width': 1.0,
            },
            RenderQuality.MEDIUM: {
                'point_size': self.config.point_size,
                'line_width': 2.0,
            },
            RenderQuality.HIGH: {
                'point_size': self.config.point_size + 1,
                'line_width': 3.0,
            },
            RenderQuality.ULTRA: {
                'point_size': self.config.point_size + 2,
                'line_width': 4.0,
            }
        }

        settings = quality_settings.get(quality, quality_settings[RenderQuality.HIGH])
        render_option.point_size = settings['point_size']
        render_option.line_width = settings['line_width']


class OccupancyVisualizer:
    """
    Professional 3D occupancy grid visualizer.

    This is the main class that orchestrates all visualization operations including
    rendering, user interaction, data processing, and export functionality.

    Features:
    - Interactive 3D visualization with Open3D
    - Support for semantic segmentation and optical flow
    - Batch processing and video generation
    - Flexible configuration and extensibility
    - Professional error handling and validation

    Example:
        ```python
        config = VisualizationConfig(
            voxel_size=(0.2, 0.2, 0.2),
            background_color=(0, 0, 0)
        )

        visualizer = OccupancyVisualizer(config=config)
        visualizer.visualize_occupancy(occupancy_data)
        ```
    """

    def __init__(self,
                 config: Optional[VisualizationConfig] = None,
                 color_map: Optional[np.ndarray] = None,
                 camera_config: Optional[CameraParameters] = None):
        """
        Initialize the occupancy visualizer.

        Args:
            config: Visualization configuration parameters
            color_map: Color mapping for semantic labels with shape (N, 3)
            camera_config: Camera configuration parameters
        """
        # Configuration
        self.config = config or VisualizationConfig()
        self.camera_config = camera_config or CameraParameters()

        # Color mapping
        if color_map is not None:
            color_map = np.array(color_map)
            if color_map.shape[1] > 3:
                color_map = color_map[:, :3]  # Take only RGB
        self.color_map = color_map

        # Component initialization
        self._initialize_components()

        # State variables
        self._reset_state()

        # Visualization objects
        self.o3d_vis = None
        self.view_control = None
        self.current_geometries = []

        logger.info("OccupancyVisualizer initialized successfully")

    def _initialize_components(self) -> None:
        """Initialize all component classes."""
        self.camera_controller = CameraController(self.camera_config)
        self.geometry_generator = GeometryGenerator()
        self.mesh_loader = MeshLoader()
        self.color_processor = ColorProcessor()
        self.voxel_processor = VoxelProcessor()
        self.render_manager = RenderManager(self.config)
        self.validator = InputValidator()
        self.event_handler = None  # Initialize when needed

    def _reset_state(self) -> None:
        """Reset internal state variables."""
        self.flag_pause = False
        self.flag_next = False
        self.flag_exit = False
        self.flag_save_on_exit = False

    def load_external_camera_parameters(self, view_json_path: Union[str, Path]) -> None:
        """
        Load external camera parameters for consistent visualization across samples.

        This method loads camera parameters from an external view.json file
        and applies them to all subsequent visualizations. This is particularly
        useful when you want to maintain the same viewing angle across different
        datasets or samples.

        Args:
            view_json_path: Path to the view.json file with camera parameters

        Example:
            # Load parameters once
            visualizer.load_external_camera_parameters('my_saved_view.json')

            # All subsequent visualizations will use these parameters
            visualizer.visualize_occupancy(data1, save_path='frame1.png')
            visualizer.visualize_occupancy(data2, save_path='frame2.png')
            # Both will have the same camera angle
        """
        self.camera_controller.load_external_parameters(view_json_path)
        logger.info(f"External camera parameters loaded from {view_json_path}")

    def has_camera_parameters(self) -> bool:
        """Check if external camera parameters are loaded."""
        return self.camera_controller.has_parameters()

    def clear_camera_parameters(self) -> None:
        """Clear any loaded external camera parameters."""
        self.camera_controller.clear_parameters()

    def get_camera_info(self) -> Optional[Dict]:
        """Get information about currently loaded camera parameters."""
        return self.camera_controller.get_parameters_info()

    @require_open3d
    def _initialize_visualizer(self) -> o3d.visualization.VisualizerWithKeyCallback:
        """Initialize Open3D visualizer with event handlers."""
        # Create visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()

        # Setup event handler
        self.event_handler = EventHandler(self)

        # Register callbacks
        for key, callback in self.event_handler.key_bindings.items():
            if key == 32:  # Space key needs special handling
                vis.register_key_action_callback(key,
                    lambda vis, action, mods: callback(vis) if action == 1 else True)
            else:
                vis.register_key_callback(key, callback)

        # Create window
        vis.create_window(
            width=self.camera_config.width,
            height=self.camera_config.height,
            window_name="3D Occupancy Visualizer"
        )

        # Setup rendering
        self.render_manager.setup_render_options(vis)

        # Get view control
        self.view_control = vis.get_view_control()

        return vis

    def visualize_occupancy(self,
                           occupancy_data: np.ndarray,
                           flow_data: Optional[np.ndarray] = None,
                           mode: VisualizationMode = VisualizationMode.SEMANTIC,
                           save_path: Optional[Union[str, Path]] = None,
                           car_model_path: Optional[Union[str, Path]] = None,
                           view_json_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Visualize single occupancy frame.

        Args:
            occupancy_data: Occupancy voxel data with shape (H, W, D)
            flow_data: Optional flow data with shape (H, W, D, 2)
            mode: Visualization mode (semantic, flow, or combined)
            save_path: Optional path to save screenshot
            car_model_path: Optional path to car model mesh
            view_json_path: Optional path to camera view parameters

        Returns:
            True if visualization should continue, False to stop
        """
        # Validate inputs
        self.validator.validate_voxel_data(occupancy_data)
        if flow_data is not None:
            self.validator.validate_flow_data(flow_data, occupancy_data.shape)

        # Process voxel data
        points, labels, flow_vectors = self.voxel_processor.voxel_to_points(
            voxel=occupancy_data,
            voxel_flow=flow_data,
            voxel_size=self.config.voxel_size,
            range_vals=self.config.range_vals,
            ignore_labels=self.config.ignore_labels
        )

        if points.shape[0] == 0:
            logger.warning("No valid voxels to visualize")
            return True

        # Determine colors based on mode
        colors = self._get_point_colors(labels, flow_vectors, mode)

        # Load car model if specified
        car_mesh = None
        if car_model_path:
            car_mesh = self.mesh_loader.load_mesh(car_model_path)

        # Generate ego car points if needed
        ego_points = None
        if self.config.show_ego_car and car_mesh is None:
            ego_points = self.geometry_generator.generate_ego_car()

        # Create geometries
        geometries = self._create_scene_geometries(
            points=points,
            colors=colors,
            ego_points=ego_points,
            car_mesh=car_mesh
        )

        # Render scene
        success = self._render_scene(
            geometries=geometries,
            save_path=save_path,
            view_json_path=view_json_path
        )

        return success and not self.flag_exit

    def _get_point_colors(self,
                         labels: np.ndarray,
                         flow_vectors: Optional[np.ndarray],
                         mode: VisualizationMode) -> np.ndarray:
        """Get point colors based on visualization mode."""
        if mode == VisualizationMode.SEMANTIC:
            if self.color_map is not None:
                colors = self.color_map[labels.astype(int) % len(self.color_map)]
            else:
                # Default rainbow colors
                colors = self._generate_default_colors(labels)

        elif mode == VisualizationMode.FLOW:
            if flow_vectors is None:
                raise OccupancyVisualizerError("Flow data required for flow visualization mode")
            vx, vy = flow_vectors[..., 0], flow_vectors[..., 1]
            colors = self.color_processor.flow_to_color(vx, vy)

        elif mode == VisualizationMode.COMBINED:
            # Combine semantic and flow information
            if flow_vectors is None:
                colors = self._get_point_colors(labels, None, VisualizationMode.SEMANTIC)
            else:
                # Use flow magnitude to modulate semantic colors
                semantic_colors = self._get_point_colors(labels, None, VisualizationMode.SEMANTIC)
                flow_magnitude = np.linalg.norm(flow_vectors, axis=1)
                max_magnitude = np.max(flow_magnitude) if flow_magnitude.size > 0 else 1.0

                if max_magnitude > 0:
                    intensity = flow_magnitude / max_magnitude
                    colors = semantic_colors * intensity[:, None]
                else:
                    colors = semantic_colors
        else:
            raise OccupancyVisualizerError(f"Unknown visualization mode: {mode}")

        return colors.astype(np.uint8)

    def _generate_default_colors(self, labels: np.ndarray) -> np.ndarray:
        """Generate default rainbow colors for labels."""
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)

        # Create rainbow color map
        colors = []
        for i, label in enumerate(unique_labels):
            hue = i / max(1, num_labels - 1)  # Avoid division by zero
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors.append([int(c * 255) for c in rgb])

        # Map labels to colors
        label_to_color = dict(zip(unique_labels, colors))
        point_colors = np.array([label_to_color[label] for label in labels])

        return point_colors

    @require_open3d
    def _create_scene_geometries(self,
                                points: np.ndarray,
                                colors: np.ndarray,
                                ego_points: Optional[np.ndarray] = None,
                                car_mesh: Optional[o3d.geometry.TriangleMesh] = None) -> List[o3d.geometry.Geometry]:
        """Create all geometries for the scene."""
        geometries = []

        # Main point cloud
        pcd = self.render_manager.create_point_cloud(points, colors, ego_points)

        # Create voxel grid for better visualization
        voxel_grid = self.render_manager.create_voxel_grid(pcd, self.config.voxel_size[0])
        geometries.append(voxel_grid)

        # Add point cloud if not using voxels
        geometries.append(pcd)

        # Bounding boxes
        if points.shape[0] > 0:
            bboxes = self.voxel_processor.get_voxel_profile(points, self.config.voxel_size)
            if bboxes.shape[0] > 0:
                bbox_corners = self.geometry_generator.compute_box_3d(
                    bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7]
                )
                line_set = self.render_manager.create_bounding_boxes(bbox_corners)
                geometries.append(line_set)

        # Coordinate frame
        if self.config.show_coordinate_frame:
            frame = self.geometry_generator.create_coordinate_frame(
                size=self.config.frame_size
            )
            geometries.append(frame)

        # Car model
        if car_mesh is not None:
            geometries.append(car_mesh)

        return geometries

    def _render_scene(self,
                     geometries: List[o3d.geometry.Geometry],
                     save_path: Optional[Union[str, Path]] = None,
                     view_json_path: Optional[Union[str, Path]] = None) -> bool:
        """Render the complete scene."""
        try:
            # Initialize visualizer if needed
            if self.o3d_vis is None:
                self.o3d_vis = self._initialize_visualizer()

            # Clear previous geometries
            self._clear_geometries()

            # Add new geometries
            for geom in geometries:
                self.o3d_vis.add_geometry(geom)
            self.current_geometries = geometries

            # Load camera parameters
            if view_json_path:
                try:
                    camera_params = self.camera_controller.load_camera_parameters(view_json_path)
                    self.camera_controller.apply_parameters(self.view_control)
                except OccupancyVisualizerError as e:
                    logger.warning(f"Could not load camera parameters: {e}")

            # Run visualization loop
            success = self._run_visualization_loop()

            # Save screenshot if requested
            if save_path and success and not self.flag_exit:
                self._save_screenshot(save_path)

            # Save camera parameters
            if not self.flag_exit:
                try:
                    self.camera_controller.save_camera_parameters(
                        self.view_control, 'view.json'
                    )
                except Exception as e:
                    logger.warning(f"Could not save camera parameters: {e}")

            return success

        except Exception as e:
            logger.error(f"Error during rendering: {e}")
            return False

    def _run_visualization_loop(self) -> bool:
        """Run the main visualization loop."""
        if self.config.wait_time > 0:
            # Timed mode
            return self._run_timed_loop()
        else:
            # Interactive mode
            return self._run_interactive_loop()

    def _run_timed_loop(self) -> bool:
        """Run timed visualization loop."""
        start_time = time.time()

        while time.time() - start_time < self.config.wait_time:
            if not self.o3d_vis.poll_events():
                self.flag_exit = True
                return False

            self.o3d_vis.update_renderer()

            # Handle pause
            while self.flag_pause:
                if not self.o3d_vis.poll_events():
                    self.flag_exit = True
                    return False
                self.o3d_vis.update_renderer()
                time.sleep(0.01)  # Small delay to prevent busy waiting

        return True

    def _run_interactive_loop(self) -> bool:
        """Run interactive visualization loop."""
        while not self.flag_next and not self.flag_exit:
            if not self.o3d_vis.poll_events():
                self.flag_exit = True
                return False

            self.o3d_vis.update_renderer()
            time.sleep(0.01)  # Small delay to prevent busy waiting

        self.flag_next = False  # Reset for next frame
        return not self.flag_exit

    def _clear_geometries(self) -> None:
        """Clear all current geometries from visualizer."""
        if self.o3d_vis is not None:
            try:
                for geom in self.current_geometries:
                    self.o3d_vis.remove_geometry(geom, reset_bounding_box=False)
            except:
                # Fallback: clear all geometries
                self.o3d_vis.clear_geometries()
        self.current_geometries = []

    def _save_screenshot(self, save_path: Union[str, Path]) -> None:
        """Save screenshot of current visualization."""
        try:
            save_path = Path(save_path)
            if save_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                save_path = save_path.with_suffix('.png')

            self.o3d_vis.capture_screen_image(str(save_path))
            logger.info(f"Screenshot saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")

    def visualize_bev(self,
                     occupancy_data: np.ndarray,
                     save_path: Optional[Union[str, Path]] = None,
                     free_class: Optional[int] = None,
                     dataset_type: str = 'nuscenes',
                     add_metadata: bool = True,
                     show_interactive: bool = True) -> bool:
        """
        Visualize occupancy data in Bird's Eye View (BEV) format.

        This method converts 3D occupancy data to a 2D top-down view and displays
        it using OpenCV. The BEV shows the topmost non-free voxel for each (x, y) location.

        Args:
            occupancy_data: 3D occupancy grid with shape (H, W, D)
            save_path: Optional path to save BEV image
            free_class: Label representing free/empty space (auto-detected if None)
            dataset_type: Dataset type for specific optimizations ('nuscenes', 'openocc')
            add_metadata: Whether to add informational text overlay
            show_interactive: Whether to show interactive OpenCV window

        Returns:
            True if visualization succeeded, False otherwise

        Example:
            # Basic BEV visualization
            success = visualizer.visualize_bev(
                occupancy_data=occupancy_grid,
                save_path="bev_view.png",
                dataset_type='nuscenes'
            )
        """
        try:
            # Auto-detect free class if not provided
            if free_class is None:
                free_class = BEVProcessor.get_dataset_free_class(dataset_type)

            logger.info(f"Creating BEV visualization: dataset={dataset_type}, free_class={free_class}")

            # Create BEV processor
            bev_processor = BEVProcessor.create_for_dataset(dataset_type, self.color_map)

            # Convert to BEV with metadata
            if add_metadata:
                bev_image, metadata = bev_processor.create_bev_with_metadata(
                    occupancy_data, free_class, self.color_map, dataset_type, False
                )
                logger.info(f"BEV created: {metadata['original_shape']} -> {metadata['bev_shape'][:2]}, "
                          f"occupancy={metadata['occupancy_ratio']:.2%}")
            else:
                bev_image = bev_processor.convert_to_bev(
                    occupancy_data, free_class, self.color_map, dataset_type
                )

            # Save image if requested
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if cv2 is not None:
                    cv2.imwrite(str(save_path), bev_image)
                    logger.info(f"BEV image saved to {save_path}")
                else:
                    logger.warning("OpenCV not available, cannot save BEV image")

            # Show interactive window if requested
            if show_interactive and cv2 is not None:
                window_name = f"BEV - {dataset_type.upper()}"
                cv2.imshow(window_name, bev_image)

                # Wait for user interaction or auto-advance
                if self.config.wait_time > 0:
                    cv2.waitKey(int(self.config.wait_time * 1000))
                else:
                    cv2.waitKey(0)  # Wait for key press

                cv2.destroyWindow(window_name)

            return True

        except Exception as e:
            logger.error(f"BEV visualization failed: {e}")
            return False

    def visualize_bev_comparison(self,
                               gt_data: np.ndarray,
                               pred_data: np.ndarray,
                               save_path: Optional[Union[str, Path]] = None,
                               free_class: Optional[int] = None,
                               dataset_type: str = 'nuscenes',
                               comparison_mode: str = 'side_by_side',
                               show_interactive: bool = True) -> bool:
        """
        Visualize BEV comparison between ground truth and prediction data.

        Args:
            gt_data: Ground truth 3D occupancy grid
            pred_data: Prediction 3D occupancy grid
            save_path: Optional path to save comparison image
            free_class: Label representing free/empty space
            dataset_type: Dataset type for specific optimizations
            comparison_mode: 'side_by_side', 'overlay', or 'difference'
            show_interactive: Whether to show interactive OpenCV window

        Returns:
            True if visualization succeeded, False otherwise
        """
        try:
            if gt_data.shape != pred_data.shape:
                raise OccupancyVisualizerError(f"Data shape mismatch: GT={gt_data.shape}, Pred={pred_data.shape}")

            # Auto-detect free class if not provided
            if free_class is None:
                free_class = BEVProcessor.get_dataset_free_class(dataset_type)

            logger.info(f"Creating BEV comparison: mode={comparison_mode}, dataset={dataset_type}")

            # Create BEV processor
            bev_processor = BEVProcessor.create_for_dataset(dataset_type, self.color_map)

            # Convert both to BEV
            gt_bev = bev_processor.convert_to_bev(gt_data, free_class, self.color_map, dataset_type)
            pred_bev = bev_processor.convert_to_bev(pred_data, free_class, self.color_map, dataset_type)

            # Create comparison based on mode
            if comparison_mode == 'side_by_side':
                comparison_image = self._create_bev_side_by_side(gt_bev, pred_bev)
            elif comparison_mode == 'overlay':
                comparison_image = self._create_bev_overlay(gt_bev, pred_bev)
            elif comparison_mode == 'difference':
                comparison_image = self._create_bev_difference(gt_data, pred_data, free_class, bev_processor)
            else:
                raise OccupancyVisualizerError(f"Unknown comparison mode: {comparison_mode}")

            # Save image if requested
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if cv2 is not None:
                    cv2.imwrite(str(save_path), comparison_image)
                    logger.info(f"BEV comparison saved to {save_path}")

            # Show interactive window if requested
            if show_interactive and cv2 is not None:
                window_name = f"BEV Comparison - {comparison_mode}"
                cv2.imshow(window_name, comparison_image)

                if self.config.wait_time > 0:
                    cv2.waitKey(int(self.config.wait_time * 1000))
                else:
                    cv2.waitKey(0)

                cv2.destroyWindow(window_name)

            return True

        except Exception as e:
            logger.error(f"BEV comparison visualization failed: {e}")
            return False

    @require_cv2
    def _create_bev_side_by_side(self, gt_bev: np.ndarray, pred_bev: np.ndarray) -> np.ndarray:
        """Create side-by-side BEV comparison."""
        # Ensure same size
        if gt_bev.shape != pred_bev.shape:
            pred_bev = cv2.resize(pred_bev, (gt_bev.shape[1], gt_bev.shape[0]))

        # Create side-by-side image
        combined = np.hstack([gt_bev, pred_bev])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Ground Truth", (10, 30), font, 1.0, (255, 255, 255), 2)
        cv2.putText(combined, "Prediction", (gt_bev.shape[1] + 10, 30), font, 1.0, (255, 255, 255), 2)

        return combined

    @require_cv2
    def _create_bev_overlay(self, gt_bev: np.ndarray, pred_bev: np.ndarray) -> np.ndarray:
        """Create overlay BEV comparison."""
        # Create overlay with transparency
        alpha = 0.7
        overlay = cv2.addWeighted(gt_bev, alpha, pred_bev, 1 - alpha, 0)

        # Add title
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, "GT + Prediction Overlay", (10, 30), font, 1.0, (255, 255, 255), 2)

        return overlay

    @require_cv2
    def _create_bev_difference(self, gt_data: np.ndarray, pred_data: np.ndarray,
                             free_class: int, bev_processor: BEVProcessor) -> np.ndarray:
        """Create difference map BEV comparison."""
        # Create difference map in 3D first
        diff_data = np.zeros_like(gt_data, dtype=np.int32)

        # Mark correct predictions
        correct_mask = (gt_data == pred_data)
        diff_data[correct_mask] = 1  # Correct class

        # Mark incorrect predictions
        incorrect_mask = (gt_data != pred_data)
        diff_data[incorrect_mask] = 2  # Incorrect class

        # Mark background/free areas
        free_mask = (gt_data == free_class)
        diff_data[free_mask] = 0  # Background

        # Create simple color map for difference visualization
        diff_colors = np.array([
            [50, 50, 50],     # Background (gray)
            [0, 255, 0],      # Correct (green)
            [0, 0, 255],      # Incorrect (red)
        ], dtype=np.uint8)

        # Convert to BEV
        diff_bev = bev_processor.convert_to_bev(diff_data, 0, diff_colors, 'nuscenes')

        # Add title and legend
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(diff_bev, "BEV Accuracy Map", (10, 30), font, 1.0, (255, 255, 255), 2)
        cv2.putText(diff_bev, "Green: Correct, Red: Incorrect", (10, diff_bev.shape[0] - 20),
                   font, 0.6, (255, 255, 255), 1)

        return diff_bev

    def get_bev_processor(self, dataset_type: str = 'nuscenes') -> BEVProcessor:
        """
        Get a BEV processor configured for this visualizer.

        Args:
            dataset_type: Dataset type for processor configuration

        Returns:
            Configured BEV processor
        """
        return BEVProcessor.create_for_dataset(dataset_type, self.color_map)

    def cleanup(self) -> None:
        """Clean up resources and close visualizer."""
        try:
            self._clear_geometries()

            if self.o3d_vis is not None:
                self.o3d_vis.destroy_window()
                self.o3d_vis = None

            self.view_control = None
            self._reset_state()

            logger.info("Visualizer cleaned up successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class BatchProcessor:
    """Handles batch processing of occupancy data sequences."""

    def __init__(self,
                 visualizer: OccupancyVisualizer,
                 output_dir: Union[str, Path] = "output"):
        """
        Initialize batch processor.

        Args:
            visualizer: OccupancyVisualizer instance
            output_dir: Directory for output files
        """
        self.visualizer = visualizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_4d_sequence(self,
                           data_4d: np.ndarray,
                           flow_4d: Optional[np.ndarray] = None,
                           scene_name: str = "4d_sequence",
                           create_video: bool = True,
                           video_fps: int = 5,
                           wait_time: float = 0.5,
                           maintain_camera: bool = True) -> List[Path]:
        """
        Process 4D occupancy sequence (time series).

        Args:
            data_4d: 4D occupancy data with shape (T, H, W, D)
            flow_4d: Optional 4D flow data with shape (T, H, W, D, 2)
            scene_name: Name for the sequence
            create_video: Whether to create video from frames
            video_fps: Video frame rate
            wait_time: Time to wait between frames (automatic advancement)
            maintain_camera: Whether to maintain camera position across frames

        Returns:
            List of generated image paths
        """
        if data_4d.ndim != 4:
            raise OccupancyVisualizerError(f"Expected 4D data, got {data_4d.ndim}D")

        num_frames = data_4d.shape[0]
        logger.info(f"Processing {num_frames} frames for scene '{scene_name}'")

        # Set visualizer wait_time for automatic frame advancement
        original_wait_time = self.visualizer.config.wait_time
        if wait_time > 0:
            self.visualizer.config.wait_time = wait_time
            logger.info(f"Automatic frame advancement enabled: {wait_time}s per frame")

        saved_images = []
        view_json_path = None

        # Use camera persistence if requested
        if maintain_camera:
            view_json_path = 'view.json'

        try:
            for frame_idx in range(num_frames):
                if self.visualizer.flag_exit:
                    logger.info(f"Processing stopped at frame {frame_idx}")
                    break

                logger.info(f"Processing frame {frame_idx + 1}/{num_frames}")

                # Get current frame data
                current_frame = data_4d[frame_idx]
                current_flow = flow_4d[frame_idx] if flow_4d is not None else None

                # Setup save path
                save_path = self.output_dir / f"{scene_name}_frame_{frame_idx:04d}.png"
                saved_images.append(save_path)

                # Visualize frame with camera persistence
                success = self.visualizer.visualize_occupancy(
                    occupancy_data=current_frame,
                    flow_data=current_flow,
                    save_path=save_path,
                    view_json_path=view_json_path  # This maintains camera position
                )

                if not success:
                    logger.warning(f"Visualization failed for frame {frame_idx}")
                    break

        finally:
            # Restore original wait_time
            self.visualizer.config.wait_time = original_wait_time

        # Create video if requested
        if create_video and saved_images:
            self._create_video(saved_images, scene_name, video_fps)

        logger.info(f"Processed {len(saved_images)} frames for scene '{scene_name}'")
        return saved_images

    def process_nuscenes_sequence(self,
                                 infos_path: Union[str, Path],
                                 data_version: str = 'v1.0-mini',
                                 data_path: Union[str, Path] = 'data/nuscenes',
                                 vis_scenes: Optional[List[str]] = None,
                                 ignore_labels: Optional[List[int]] = None,
                                 car_model_mesh: Optional[Union[str, Path]] = None,
                                 use_car_model: bool = False,
                                 create_videos: bool = True,
                                 video_fps: int = 5,
                                 wait_time: float = 0.5,
                                 maintain_camera: bool = True,
                                 voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
                                 range_vals: Tuple[float, ...] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)) -> Dict[str, List[Path]]:
        """
        Process NuScenes occupancy sequences from PKL files.

        This method reads NuScenes info PKL files, organizes data by scenes,
        and sequentially visualizes 3D occupancy data for each scene. It matches
        the original vis_sequential_occ functionality with enhanced features.

        Args:
            infos_path: Path to NuScenes info pickle file (e.g., nuscenes_infos_val.pkl)
            data_version: NuScenes data version ('v1.0-mini', 'v1.0-trainval', etc.)
            data_path: Path to NuScenes data directory
            vis_scenes: List of specific scene names to visualize (None for all scenes)
            ignore_labels: List of semantic labels to ignore during visualization
            car_model_mesh: Path to car model mesh file (.obj) for ego vehicle
            use_car_model: Whether to show ego vehicle model
            create_videos: Whether to create videos for each scene
            video_fps: Video frame rate
            wait_time: Time to wait between frames (automatic advancement)
            maintain_camera: Whether to maintain camera position across frames
            voxel_size: Voxel size for occupancy grid (x, y, z)
            range_vals: Spatial range for occupancy grid (xmin, ymin, zmin, xmax, ymax, zmax)

        Returns:
            Dictionary mapping scene names to lists of generated image paths

        Example:
            # Visualize specific NuScenes scenes
            processor = BatchProcessor(visualizer, "nuscenes_output")
            results = processor.process_nuscenes_sequence(
                infos_path="nuscenes_infos_val.pkl",
                data_version="v1.0-mini",
                data_path="data/nuscenes",
                vis_scenes=["scene-0061", "scene-0103"],
                wait_time=0.5,
                maintain_camera=True
            )
        """
        try:
            import pickle
            from nuscenes import NuScenes
        except ImportError:
            raise OccupancyVisualizerError(
                "NuScenes processing requires nuscenes-devkit. "
                "Install with: pip install nuscenes-devkit"
            )

        # Set default ignore labels if not provided
        if ignore_labels is None:
            ignore_labels = [0, 17]  # Background and undefined

        # Set visualizer configuration for NuScenes processing
        original_wait_time = self.visualizer.config.wait_time
        original_ignore_labels = self.visualizer.config.ignore_labels

        # Apply NuScenes-specific settings
        self.visualizer.config.wait_time = wait_time if wait_time > 0 else -1
        self.visualizer.config.ignore_labels = ignore_labels

        # Update spatial configuration
        self.visualizer.config.voxel_size = voxel_size
        self.visualizer.config.range_vals = range_vals

        logger.info(f"Loading NuScenes data from {infos_path}")

        try:
            # Load data
            with open(infos_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'infos' in data:
                    infos = data['infos']
                else:
                    infos = data  # Direct list format

            logger.info(f"Loaded {len(infos)} samples from PKL file")

            nusc = NuScenes(data_version, str(data_path))
            scenes_data = self._organize_by_scene(infos, nusc, vis_scenes)

            logger.info(f"Found {len(scenes_data)} scenes to process")

            all_results = {}

            # Process each scene
            for scene_name, scene_infos in scenes_data.items():
                logger.info(f"Processing scene: {scene_name} ({len(scene_infos)} frames)")

                saved_images = []
                should_continue = True

                # Set up camera persistence for the scene
                view_json_path = None
                if maintain_camera:
                    view_json_path = 'view.json'

                for idx, info in enumerate(scene_infos):
                    if self.visualizer.flag_exit or not should_continue:
                        logger.info(f"Processing stopped for scene {scene_name} at frame {idx}")
                        break

                    logger.info(f"Processing {scene_name}, frame {idx + 1}/{len(scene_infos)}")

                    try:
                        # Load occupancy data
                        occ_path = Path(info['occ_path']) / 'labels.npz'
                        if not occ_path.exists():
                            logger.warning(f"Occupancy data not found: {occ_path}")
                            continue

                        occupancy_data = np.load(occ_path)['semantics']
                        logger.debug(f"Loaded occupancy data shape: {occupancy_data.shape}")

                        # Setup save path
                        save_path = self.output_dir / f"{scene_name}_{idx:04d}.png"
                        saved_images.append(save_path)

                        # Visualize frame with all NuScenes-specific parameters
                        success = self.visualizer.visualize_occupancy(
                            occupancy_data=occupancy_data,
                            save_path=save_path,
                            car_model_path=car_model_mesh,
                            view_json_path=view_json_path  # Maintain camera consistency
                        )

                        if not success:
                            logger.warning(f"Visualization failed for {scene_name}, frame {idx}")
                            should_continue = False
                            break

                    except Exception as e:
                        logger.error(f"Error processing frame {idx} of scene {scene_name}: {e}")
                        continue

                # Create video for scene
                if create_videos and saved_images:
                    logger.info(f"Creating video for scene {scene_name}")
                    video_path = self._create_video(saved_images, scene_name, video_fps)
                    if video_path:
                        logger.info(f"Video created: {video_path}")

                all_results[scene_name] = saved_images
                logger.info(f"Completed scene {scene_name}: {len(saved_images)} frames processed")

        finally:
            # Restore original settings
            self.visualizer.config.wait_time = original_wait_time
            self.visualizer.config.ignore_labels = original_ignore_labels

        logger.info(f"NuScenes processing completed. Total scenes: {len(all_results)}")
        return all_results

    def _organize_by_scene(self,
                          infos: List[Dict],
                          nusc,
                          vis_scenes: Optional[List[str]]) -> Dict[str, List[Dict]]:
        """Organize info data by scene."""
        scenes = {}

        for info in infos:
            scene_token = nusc.get('sample', info['token'])['scene_token']
            scene_meta = nusc.get('scene', scene_token)
            scene_name = scene_meta['name']

            if scene_name not in scenes:
                scenes[scene_name] = []
            scenes[scene_name].append(info)

        # Filter scenes if specified
        if vis_scenes:
            filtered_scenes = {}
            for scene_name in vis_scenes:
                if scene_name in scenes:
                    filtered_scenes[scene_name] = scenes[scene_name]
                else:
                    logger.warning(f"Scene '{scene_name}' not found in data")
            return filtered_scenes

        return scenes

    @require_cv2
    def _create_video(self,
                     image_paths: List[Path],
                     scene_name: str,
                     fps: int = 5) -> Optional[Path]:
        """Create video from image sequence."""
        if not image_paths:
            logger.warning("No images to create video")
            return None

        # Check if first image exists and get dimensions
        first_image_path = image_paths[0]
        if not first_image_path.exists():
            logger.error(f"First image not found: {first_image_path}")
            return None

        try:
            first_image = cv2.imread(str(first_image_path))
            if first_image is None:
                logger.error(f"Could not read first image: {first_image_path}")
                return None

            height, width = first_image.shape[:2]

            # Setup video writer
            video_path = self.output_dir / f"{scene_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(video_path), fourcc, fps, (width, height)
            )

            # Write frames
            frames_written = 0
            for img_path in image_paths:
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        video_writer.write(img)
                        frames_written += 1
                    else:
                        logger.warning(f"Could not read image: {img_path}")
                else:
                    logger.warning(f"Image not found: {img_path}")

            video_writer.release()

            logger.info(f"Video created: {video_path} ({frames_written} frames)")
            return video_path

        except Exception as e:
            logger.error(f"Failed to create video: {e}")
            return None

    def process_nuscenes_predictions(self,
                                   infos_path: Union[str, Path],
                                   pred_path: Union[str, Path],
                                   data_version: str = 'v1.0-mini',
                                   data_path: Union[str, Path] = 'data/nuscenes',
                                   vis_scenes: Optional[List[str]] = None,
                                   ignore_labels: Optional[List[int]] = None,
                                   comparison_mode: str = 'side_by_side',
                                   car_model_mesh: Optional[Union[str, Path]] = None,
                                   use_car_model: bool = False,
                                   create_videos: bool = True,
                                   video_fps: int = 5,
                                   wait_time: float = 0.5,
                                   maintain_camera: bool = True,
                                   voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
                                   range_vals: Tuple[float, ...] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)) -> Dict[str, Dict[str, List[Path]]]:
        """
        Process NuScenes occupancy predictions alongside ground truth for comparison.

        This method visualizes both GT occupancy data (from PKL files) and prediction data,
        supporting side-by-side comparison or separate processing modes.

        File structure expected:
            GT: data/nuscenes/gts/scene_name/sample_id/labels.npz (from PKL occ_path)
            Prediction: pred_path/scene_name/sample_id/labels.npz

        Args:
            infos_path: Path to NuScenes info pickle file (e.g., nuscenes_infos_val.pkl)
            pred_path: Root path to prediction files (e.g., 'pred')
            data_version: NuScenes data version ('v1.0-mini', 'v1.0-trainval', etc.)
            data_path: Path to NuScenes data directory
            vis_scenes: List of specific scene names to visualize (None for all scenes)
            ignore_labels: List of semantic labels to ignore during visualization
            comparison_mode: 'side_by_side', 'separate', 'overlay', or 'difference'
            car_model_mesh: Path to car model mesh file (.obj) for ego vehicle
            use_car_model: Whether to show ego vehicle model
            create_videos: Whether to create videos for each scene
            video_fps: Video frame rate
            wait_time: Time to wait between frames (automatic advancement)
            maintain_camera: Whether to maintain camera position across frames
            voxel_size: Voxel size for occupancy grid (x, y, z)
            range_vals: Spatial range for occupancy grid (xmin, ymin, zmin, xmax, ymax, zmax)

        Returns:
            Dictionary with structure: {scene_name: {'gt': [gt_paths], 'pred': [pred_paths], 'comparison': [comp_paths]}}

        Example:
            # Compare GT vs predictions for specific scenes
            processor = BatchProcessor(visualizer, "prediction_comparison")
            results = processor.process_nuscenes_predictions(
                infos_path="nuscenes_infos_val.pkl",
                pred_path="pred",
                vis_scenes=["scene-0061", "scene-0103"],
                comparison_mode='side_by_side',
                wait_time=0.5
            )
        """
        try:
            import pickle
            from nuscenes import NuScenes
        except ImportError:
            raise OccupancyVisualizerError(
                "NuScenes processing requires nuscenes-devkit. "
                "Install with: pip install nuscenes-devkit"
            )

        pred_path = Path(pred_path)
        if not pred_path.exists():
            raise OccupancyVisualizerError(f"Prediction path does not exist: {pred_path}")

        # Set default ignore labels if not provided
        if ignore_labels is None:
            ignore_labels = [0, 17]  # Background and undefined

        # Set visualizer configuration for NuScenes processing
        original_wait_time = self.visualizer.config.wait_time
        original_ignore_labels = self.visualizer.config.ignore_labels

        # Apply NuScenes-specific settings
        self.visualizer.config.wait_time = wait_time if wait_time > 0 else -1
        self.visualizer.config.ignore_labels = ignore_labels

        # Update spatial configuration
        self.visualizer.config.voxel_size = voxel_size
        self.visualizer.config.range_vals = range_vals

        logger.info(f"Loading NuScenes data from {infos_path}")
        logger.info(f"Prediction data path: {pred_path}")
        logger.info(f"Comparison mode: {comparison_mode}")

        try:
            # Load data
            with open(infos_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'infos' in data:
                    infos = data['infos']
                else:
                    infos = data  # Direct list format

            logger.info(f"Loaded {len(infos)} samples from PKL file")

            nusc = NuScenes(data_version, str(data_path))
            scenes_data = self._organize_by_scene(infos, nusc, vis_scenes)

            logger.info(f"Found {len(scenes_data)} scenes to process")

            all_results = {}

            # Process each scene
            for scene_name, scene_infos in scenes_data.items():
                logger.info(f"Processing scene: {scene_name} ({len(scene_infos)} frames)")

                scene_results = {
                    'gt': [],
                    'pred': [],
                    'comparison': []
                }
                should_continue = True

                # Set up camera persistence for the scene
                view_json_path = None
                if maintain_camera:
                    view_json_path = 'view.json'

                for idx, info in enumerate(scene_infos):
                    if self.visualizer.flag_exit or not should_continue:
                        logger.info(f"Processing stopped for scene {scene_name} at frame {idx}")
                        break

                    logger.info(f"Processing {scene_name}, frame {idx + 1}/{len(scene_infos)}")

                    try:
                        # Get sample token and construct paths
                        sample_token = info['token']

                        # GT path (from occ_path in PKL)
                        gt_occ_path = Path(info['occ_path']) / 'labels.npz'

                        # Prediction path (construct from pred_path structure)
                        # Extract scene and sample info for prediction path
                        pred_scene_path = pred_path / scene_name / sample_token / 'labels.npz'

                        # Check if both files exist
                        gt_exists = gt_occ_path.exists()
                        pred_exists = pred_scene_path.exists()

                        if not gt_exists:
                            logger.warning(f"GT occupancy data not found: {gt_occ_path}")
                        if not pred_exists:
                            logger.warning(f"Prediction data not found: {pred_scene_path}")

                        if not (gt_exists and pred_exists):
                            logger.warning(f"Skipping frame {idx} - missing data files")
                            continue

                        # Load occupancy data
                        gt_data = np.load(gt_occ_path)['semantics']
                        pred_data = np.load(pred_scene_path)['semantics']

                        logger.debug(f"GT data shape: {gt_data.shape}, Pred data shape: {pred_data.shape}")

                        # Validate data shapes match
                        if gt_data.shape != pred_data.shape:
                            logger.error(f"Data shape mismatch - GT: {gt_data.shape}, Pred: {pred_data.shape}")
                            continue

                        # Process based on comparison mode
                        if comparison_mode == 'side_by_side':
                            # Create side-by-side comparison
                            success = self._process_side_by_side_comparison(
                                gt_data, pred_data, scene_name, idx,
                                scene_results, view_json_path, car_model_mesh
                            )
                        elif comparison_mode == 'separate':
                            # Process GT and predictions separately
                            success = self._process_separate_comparison(
                                gt_data, pred_data, scene_name, idx,
                                scene_results, view_json_path, car_model_mesh
                            )
                        elif comparison_mode == 'overlay':
                            # Overlay predictions on GT
                            success = self._process_overlay_comparison(
                                gt_data, pred_data, scene_name, idx,
                                scene_results, view_json_path, car_model_mesh
                            )
                        elif comparison_mode == 'difference':
                            # Show difference map
                            success = self._process_difference_comparison(
                                gt_data, pred_data, scene_name, idx,
                                scene_results, view_json_path, car_model_mesh
                            )
                        else:
                            logger.error(f"Unknown comparison mode: {comparison_mode}")
                            success = False

                        if not success:
                            logger.warning(f"Comparison failed for {scene_name}, frame {idx}")
                            should_continue = False
                            break

                    except Exception as e:
                        logger.error(f"Error processing frame {idx} of scene {scene_name}: {e}")
                        continue

                # Create videos for scene
                if create_videos:
                    self._create_comparison_videos(scene_results, scene_name, video_fps, comparison_mode)

                all_results[scene_name] = scene_results
                logger.info(f"Completed scene {scene_name}: GT={len(scene_results['gt'])}, "
                          f"Pred={len(scene_results['pred'])}, Comp={len(scene_results['comparison'])}")

        finally:
            # Restore original settings
            self.visualizer.config.wait_time = original_wait_time
            self.visualizer.config.ignore_labels = original_ignore_labels

        logger.info(f"NuScenes prediction processing completed. Total scenes: {len(all_results)}")
        return all_results

    def _process_side_by_side_comparison(self, gt_data: np.ndarray, pred_data: np.ndarray,
                                       scene_name: str, idx: int, scene_results: Dict,
                                       view_json_path: Optional[str], car_model_mesh: Optional[str]) -> bool:
        """Process side-by-side GT vs prediction comparison."""
        try:
            # Setup paths
            gt_path = self.output_dir / f"{scene_name}_{idx:04d}_gt.png"
            pred_path = self.output_dir / f"{scene_name}_{idx:04d}_pred.png"
            comparison_path = self.output_dir / f"{scene_name}_{idx:04d}_comparison.png"

            # Visualize GT
            gt_success = self.visualizer.visualize_occupancy(
                occupancy_data=gt_data,
                save_path=gt_path,
                car_model_path=car_model_mesh,
                view_json_path=view_json_path
            )

            # Visualize prediction with same camera
            pred_success = self.visualizer.visualize_occupancy(
                occupancy_data=pred_data,
                save_path=pred_path,
                car_model_path=car_model_mesh,
                view_json_path=view_json_path
            )

            if gt_success and pred_success:
                # Create side-by-side image
                comparison_success = self._create_side_by_side_image(
                    gt_path, pred_path, comparison_path,
                    f"GT vs Pred - {scene_name} Frame {idx}"
                )

                if comparison_success:
                    scene_results['gt'].append(gt_path)
                    scene_results['pred'].append(pred_path)
                    scene_results['comparison'].append(comparison_path)
                    return True

            return False

        except Exception as e:
            logger.error(f"Side-by-side comparison failed: {e}")
            return False

    def _process_separate_comparison(self, gt_data: np.ndarray, pred_data: np.ndarray,
                                   scene_name: str, idx: int, scene_results: Dict,
                                   view_json_path: Optional[str], car_model_mesh: Optional[str]) -> bool:
        """Process separate GT and prediction visualizations."""
        try:
            # Setup paths
            gt_path = self.output_dir / f"{scene_name}_{idx:04d}_gt.png"
            pred_path = self.output_dir / f"{scene_name}_{idx:04d}_pred.png"

            # Visualize GT
            gt_success = self.visualizer.visualize_occupancy(
                occupancy_data=gt_data,
                save_path=gt_path,
                car_model_path=car_model_mesh,
                view_json_path=view_json_path
            )

            # Visualize prediction
            pred_success = self.visualizer.visualize_occupancy(
                occupancy_data=pred_data,
                save_path=pred_path,
                car_model_path=car_model_mesh,
                view_json_path=view_json_path
            )

            if gt_success and pred_success:
                scene_results['gt'].append(gt_path)
                scene_results['pred'].append(pred_path)
                return True

            return False

        except Exception as e:
            logger.error(f"Separate comparison failed: {e}")
            return False

    def _process_overlay_comparison(self, gt_data: np.ndarray, pred_data: np.ndarray,
                                  scene_name: str, idx: int, scene_results: Dict,
                                  view_json_path: Optional[str], car_model_mesh: Optional[str]) -> bool:
        """Process overlay comparison (predictions overlaid on GT)."""
        try:
            # Create overlay data - show GT as base with prediction highlights
            overlay_data = gt_data.copy()

            # Highlight differences (where prediction differs from GT)
            diff_mask = (gt_data != pred_data)
            overlay_data[diff_mask] = pred_data[diff_mask] + 100  # Offset for visual distinction

            overlay_path = self.output_dir / f"{scene_name}_{idx:04d}_overlay.png"

            success = self.visualizer.visualize_occupancy(
                occupancy_data=overlay_data,
                save_path=overlay_path,
                car_model_path=car_model_mesh,
                view_json_path=view_json_path
            )

            if success:
                scene_results['comparison'].append(overlay_path)
                return True

            return False

        except Exception as e:
            logger.error(f"Overlay comparison failed: {e}")
            return False

    def _process_difference_comparison(self, gt_data: np.ndarray, pred_data: np.ndarray,
                                     scene_name: str, idx: int, scene_results: Dict,
                                     view_json_path: Optional[str], car_model_mesh: Optional[str]) -> bool:
        """Process difference map comparison."""
        try:
            # Create difference map
            diff_data = np.zeros_like(gt_data)

            # Mark correct predictions (same as GT)
            correct_mask = (gt_data == pred_data)
            diff_data[correct_mask] = 1  # Correct class

            # Mark incorrect predictions
            incorrect_mask = (gt_data != pred_data)
            diff_data[incorrect_mask] = 2  # Incorrect class

            # Mark background/ignored areas
            ignore_mask = np.isin(gt_data, self.visualizer.config.ignore_labels)
            diff_data[ignore_mask] = 0  # Background/ignored

            diff_path = self.output_dir / f"{scene_name}_{idx:04d}_difference.png"

            success = self.visualizer.visualize_occupancy(
                occupancy_data=diff_data,
                save_path=diff_path,
                car_model_path=car_model_mesh,
                view_json_path=view_json_path
            )

            if success:
                scene_results['comparison'].append(diff_path)
                return True

            return False

        except Exception as e:
            logger.error(f"Difference comparison failed: {e}")
            return False

    @require_cv2
    def _create_side_by_side_image(self, gt_path: Path, pred_path: Path,
                                 output_path: Path, title: str) -> bool:
        """Create side-by-side comparison image."""
        try:
            gt_img = cv2.imread(str(gt_path))
            pred_img = cv2.imread(str(pred_path))

            if gt_img is None or pred_img is None:
                logger.error("Failed to load images for side-by-side comparison")
                return False

            # Ensure same size
            if gt_img.shape != pred_img.shape:
                # Resize to match GT
                pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

            # Create side-by-side image
            combined_img = np.hstack([gt_img, pred_img])

            # Add title and labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (255, 255, 255)
            thickness = 2

            # Add main title
            title_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
            title_x = (combined_img.shape[1] - title_size[0]) // 2
            cv2.putText(combined_img, title, (title_x, 30), font, font_scale, color, thickness)

            # Add GT and Pred labels
            gt_label_x = gt_img.shape[1] // 2 - 50
            pred_label_x = gt_img.shape[1] + pred_img.shape[1] // 2 - 80
            cv2.putText(combined_img, "Ground Truth", (gt_label_x, 60), font, 0.8, color, 2)
            cv2.putText(combined_img, "Prediction", (pred_label_x, 60), font, 0.8, color, 2)

            # Save combined image
            cv2.imwrite(str(output_path), combined_img)
            return True

        except Exception as e:
            logger.error(f"Failed to create side-by-side image: {e}")
            return False

    @require_cv2
    def _create_comparison_videos(self, scene_results: Dict, scene_name: str,
                                fps: int, comparison_mode: str) -> None:
        """Create videos for comparison results."""
        try:
            if comparison_mode == 'side_by_side' and scene_results['comparison']:
                self._create_video(scene_results['comparison'], f"{scene_name}_comparison", fps)
            elif comparison_mode == 'separate':
                if scene_results['gt']:
                    self._create_video(scene_results['gt'], f"{scene_name}_gt", fps)
                if scene_results['pred']:
                    self._create_video(scene_results['pred'], f"{scene_name}_pred", fps)
            elif comparison_mode in ['overlay', 'difference'] and scene_results['comparison']:
                self._create_video(scene_results['comparison'], f"{scene_name}_{comparison_mode}", fps)

        except Exception as e:
            logger.error(f"Failed to create comparison videos: {e}")

    def process_bev_sequence(self,
                           occupancy_4d: np.ndarray,
                           free_class: Optional[int] = None,
                           dataset_type: str = 'nuscenes',
                           create_video: bool = True,
                           video_fps: int = 5) -> List[Path]:
        """
        Process 4D occupancy sequence into BEV format.

        Args:
            data_path: Path to 4D occupancy data file (.npz)
            free_class: Label representing free/empty space (auto-detected if None)
            dataset_type: Dataset type for specific optimizations
            create_video: Whether to create video from BEV sequence
            video_fps: Video frame rate

        Returns:
            List of paths to generated BEV images
        """
        try:
            if occupancy_4d.ndim != 4:
                raise OccupancyVisualizerError(f"Expected 4D data, got shape: {occupancy_4d.shape}")

            num_frames = occupancy_4d.shape[0]
            logger.info(f"Processing {num_frames} frames in BEV format")

            # Auto-detect free class if not provided
            if free_class is None:
                free_class = BEVProcessor.get_dataset_free_class(dataset_type)

            saved_images = []

            for frame_idx in range(num_frames):
                if self.visualizer.flag_exit:
                    logger.info(f"Processing stopped at frame {frame_idx}")
                    break

                logger.info(f"Processing BEV frame {frame_idx + 1}/{num_frames}")

                try:
                    frame_data = occupancy_4d[frame_idx]
                    save_path = self.output_dir / f"bev_frame_{frame_idx:04d}.png"

                    # Visualize frame in BEV format
                    success = self.visualizer.visualize_bev(
                        occupancy_data=frame_data,
                        save_path=save_path,
                        free_class=free_class,
                        dataset_type=dataset_type,
                        show_interactive=False  # Don't show interactive for batch processing
                    )

                    if success:
                        saved_images.append(save_path)
                    else:
                        logger.warning(f"BEV visualization failed for frame {frame_idx}")

                except Exception as e:
                    logger.error(f"Error processing BEV frame {frame_idx}: {e}")
                    continue

            logger.info(f"BEV sequence processing completed: {len(saved_images)}/{num_frames} frames")

            # Create video if requested
            if create_video and saved_images:
                logger.info("Creating BEV sequence video...")
                video_path = self._create_video(saved_images, "bev_sequence", video_fps)
                if video_path:
                    logger.info(f"BEV video created: {video_path}")

            return saved_images

        except Exception as e:
            logger.error(f"BEV sequence processing failed: {e}")
            return []

    def process_nuscenes_bev_sequence(self,
                                    infos_path: Union[str, Path],
                                    data_version: str = 'v1.0-mini',
                                    data_path: Union[str, Path] = 'data/nuscenes',
                                    vis_scenes: Optional[List[str]] = None,
                                    free_class: Optional[int] = None,
                                    dataset_type: str = 'nuscenes',
                                    create_videos: bool = True,
                                    video_fps: int = 5) -> Dict[str, List[Path]]:
        """
        Process NuScenes occupancy sequences into BEV format.

        Args:
            infos_path: Path to NuScenes info pickle file
            data_version: NuScenes data version
            data_path: Path to NuScenes data directory
            vis_scenes: List of specific scene names to visualize
            free_class: Label representing free/empty space
            dataset_type: Dataset type for BEV processing
            create_videos: Whether to create videos for each scene
            video_fps: Video frame rate

        Returns:
            Dictionary mapping scene names to lists of BEV image paths
        """
        try:
            import pickle
            from nuscenes import NuScenes
        except ImportError:
            raise OccupancyVisualizerError(
                "NuScenes BEV processing requires nuscenes-devkit. "
                "Install with: pip install nuscenes-devkit"
            )

        # Auto-detect free class if not provided
        if free_class is None:
            free_class = BEVProcessor.get_dataset_free_class(dataset_type)

        logger.info(f"Processing NuScenes BEV sequences: dataset={dataset_type}, free_class={free_class}")

        try:
            # Load data
            with open(infos_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'infos' in data:
                    infos = data['infos']
                else:
                    infos = data

            logger.info(f"Loaded {len(infos)} samples from PKL file")

            nusc = NuScenes(data_version, str(data_path))
            scenes_data = self._organize_by_scene(infos, nusc, vis_scenes)

            logger.info(f"Found {len(scenes_data)} scenes for BEV processing")

            all_results = {}

            # Process each scene in BEV format
            for scene_name, scene_infos in scenes_data.items():
                logger.info(f"Processing BEV scene: {scene_name} ({len(scene_infos)} frames)")

                saved_images = []

                for idx, info in enumerate(scene_infos):
                    if self.visualizer.flag_exit:
                        logger.info(f"BEV processing stopped for scene {scene_name} at frame {idx}")
                        break

                    logger.info(f"Processing BEV {scene_name}, frame {idx + 1}/{len(scene_infos)}")

                    try:
                        # Load occupancy data
                        occ_path = Path(info['occ_path']) / 'labels.npz'
                        if not occ_path.exists():
                            logger.warning(f"Occupancy data not found: {occ_path}")
                            continue

                        occupancy_data = np.load(occ_path)['semantics']

                        # Create BEV visualization
                        save_path = self.output_dir / f"{scene_name}_bev_{idx:04d}.png"

                        success = self.visualizer.visualize_bev(
                            occupancy_data=occupancy_data,
                            save_path=save_path,
                            free_class=free_class,
                            dataset_type=dataset_type,
                            show_interactive=False
                        )

                        if success:
                            saved_images.append(save_path)
                        else:
                            logger.warning(f"BEV visualization failed for {scene_name}, frame {idx}")

                    except Exception as e:
                        logger.error(f"Error processing BEV frame {idx} of scene {scene_name}: {e}")
                        continue

                # Create video for scene
                if create_videos and saved_images:
                    logger.info(f"Creating BEV video for scene {scene_name}")
                    video_path = self._create_video(saved_images, f"{scene_name}_bev", video_fps)
                    if video_path:
                        logger.info(f"BEV video created: {video_path}")

                all_results[scene_name] = saved_images
                logger.info(f"Completed BEV scene {scene_name}: {len(saved_images)} frames processed")

            logger.info(f"NuScenes BEV processing completed. Total scenes: {len(all_results)}")
            return all_results

        except Exception as e:
            logger.error(f"NuScenes BEV processing failed: {e}")
            return {}

    def process_bev_predictions(self,
                              infos_path: Union[str, Path],
                              pred_path: Union[str, Path],
                              data_version: str = 'v1.0-mini',
                              data_path: Union[str, Path] = 'data/nuscenes',
                              vis_scenes: Optional[List[str]] = None,
                              free_class: Optional[int] = None,
                              dataset_type: str = 'nuscenes',
                              comparison_mode: str = 'side_by_side',
                              create_videos: bool = True,
                              video_fps: int = 5) -> Dict[str, Dict[str, List[Path]]]:
        """
        Process NuScenes predictions vs GT in BEV format.

        Args:
            infos_path: Path to NuScenes info pickle file
            pred_path: Root path to prediction files
            data_version: NuScenes data version
            data_path: Path to NuScenes data directory
            vis_scenes: List of specific scene names to visualize
            free_class: Label representing free/empty space
            dataset_type: Dataset type for BEV processing
            comparison_mode: BEV comparison mode ('side_by_side', 'overlay', 'difference')
            create_videos: Whether to create videos for each scene
            video_fps: Video frame rate

        Returns:
            Dictionary with structure: {scene_name: {'gt_bev': [paths], 'pred_bev': [paths], 'comparison_bev': [paths]}}
        """
        try:
            import pickle
            from nuscenes import NuScenes
        except ImportError:
            raise OccupancyVisualizerError(
                "NuScenes BEV processing requires nuscenes-devkit"
            )

        pred_path = Path(pred_path)
        if not pred_path.exists():
            raise OccupancyVisualizerError(f"Prediction path does not exist: {pred_path}")

        # Auto-detect free class if not provided
        if free_class is None:
            free_class = BEVProcessor.get_dataset_free_class(dataset_type)

        logger.info(f"Processing NuScenes BEV predictions: mode={comparison_mode}, dataset={dataset_type}")

        try:
            # Load data
            with open(infos_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'infos' in data:
                    infos = data['infos']
                else:
                    infos = data

            nusc = NuScenes(data_version, str(data_path))
            scenes_data = self._organize_by_scene(infos, nusc, vis_scenes)

            logger.info(f"Found {len(scenes_data)} scenes for BEV prediction processing")

            all_results = {}

            # Process each scene
            for scene_name, scene_infos in scenes_data.items():
                logger.info(f"Processing BEV predictions for scene: {scene_name}")

                scene_results = {
                    'gt_bev': [],
                    'pred_bev': [],
                    'comparison_bev': []
                }

                for idx, info in enumerate(scene_infos):
                    if self.visualizer.flag_exit:
                        break

                    try:
                        # Get sample token and paths
                        sample_token = info['token']
                        gt_occ_path = Path(info['occ_path']) / 'labels.npz'
                        pred_scene_path = pred_path / scene_name / sample_token / 'labels.npz'

                        # Check if both files exist
                        if not (gt_occ_path.exists() and pred_scene_path.exists()):
                            logger.warning(f"Missing data files for frame {idx}")
                            continue

                        # Load data
                        gt_data = np.load(gt_occ_path)['semantics']
                        pred_data = np.load(pred_scene_path)['semantics']

                        if gt_data.shape != pred_data.shape:
                            logger.error(f"Shape mismatch - GT: {gt_data.shape}, Pred: {pred_data.shape}")
                            continue

                        # Create BEV comparison
                        comparison_path = self.output_dir / f"{scene_name}_bev_comparison_{idx:04d}.png"

                        success = self.visualizer.visualize_bev_comparison(
                            gt_data=gt_data,
                            pred_data=pred_data,
                            save_path=comparison_path,
                            free_class=free_class,
                            dataset_type=dataset_type,
                            comparison_mode=comparison_mode,
                            show_interactive=False
                        )

                        if success:
                            scene_results['comparison_bev'].append(comparison_path)

                    except Exception as e:
                        logger.error(f"Error processing BEV prediction frame {idx}: {e}")
                        continue

                # Create videos
                if create_videos and scene_results['comparison_bev']:
                    video_path = self._create_video(
                        scene_results['comparison_bev'],
                        f"{scene_name}_bev_{comparison_mode}",
                        video_fps
                    )
                    if video_path:
                        logger.info(f"BEV comparison video: {video_path}")

                all_results[scene_name] = scene_results
                logger.info(f"Completed BEV predictions for {scene_name}: {len(scene_results['comparison_bev'])} comparisons")

            return all_results

        except Exception as e:
            logger.error(f"BEV prediction processing failed: {e}")
            return {}


class VisualizerFactory:
    """Factory class for creating configured visualizers."""

    @staticmethod
    def create_default_visualizer(**kwargs) -> OccupancyVisualizer:
        """Create visualizer with default configuration."""
        config = VisualizationConfig()
        return OccupancyVisualizer(config=config, **kwargs)

    @staticmethod
    def create_occ3d_visualizer(**kwargs) -> OccupancyVisualizer:
        """Create visualizer with OCC3D standard color map."""
        config = VisualizationConfig()
        color_map = get_occ3d_color_map()
        return OccupancyVisualizer(config=config, color_map=color_map, **kwargs)

    @staticmethod
    def create_openocc_visualizer(**kwargs) -> OccupancyVisualizer:
        """Create visualizer with OpenOCC standard color map."""
        config = VisualizationConfig(ignore_labels=[16])
        color_map = get_openocc_color_map()
        return OccupancyVisualizer(config=config, color_map=color_map, **kwargs)

    @staticmethod
    def create_flow_visualizer(**kwargs) -> OccupancyVisualizer:
        """Create visualizer optimized for flow visualization."""
        config = VisualizationConfig(
            background_color=(0, 0, 0),  # Black background for flow
            point_size=6,
            show_coordinate_frame=True
        )
        return OccupancyVisualizer(config=config, **kwargs)

    @staticmethod
    def create_high_quality_visualizer(color_map_name: str = 'occ3d', **kwargs) -> OccupancyVisualizer:
        """
        Create visualizer with high-quality settings.

        Args:
            color_map_name: Name of color map to use ('occ3d', 'openocc', 'default')
            **kwargs: Additional arguments passed to OccupancyVisualizer
        """
        config = VisualizationConfig(
            point_size=8,
            voxel_size=(0.2, 0.2, 0.2),  # Smaller voxels for detail
            background_color=(240, 240, 240),  # Light gray
            show_coordinate_frame=True,
            frame_size=2.0
        )
        camera_config = CameraParameters(
            width=2560,
            height=1440
        )
        color_map = get_color_map_by_name(color_map_name)
        return OccupancyVisualizer(
            config=config,
            camera_config=camera_config,
            color_map=color_map,
            **kwargs
        )

    @staticmethod
    def create_with_external_camera(color_map_name: str = 'occ3d',
                                   view_json_path: Union[str, Path] = None,
                                   **kwargs) -> OccupancyVisualizer:
        """
        Create visualizer with external camera parameters pre-loaded.

        This factory method creates a visualizer and automatically loads
        external camera parameters, making it easy to maintain consistent
        viewing angles across multiple samples.

        Args:
            color_map_name: Name of color map to use ('occ3d', 'openocc', 'default')
            view_json_path: Path to external view.json file with camera parameters
            **kwargs: Additional arguments passed to OccupancyVisualizer

        Returns:
            Configured visualizer with pre-loaded camera parameters

        Example:
            # Create visualizer with consistent camera angle
            visualizer = VisualizerFactory.create_with_external_camera(
                color_map_name='occ3d',
                view_json_path='my_camera_view.json'
            )

            # All visualizations will use the same camera angle
            visualizer.visualize_occupancy(data1, save_path='sample1.png')
            visualizer.visualize_occupancy(data2, save_path='sample2.png')
        """
        config = VisualizationConfig()
        color_map = get_color_map_by_name(color_map_name)

        visualizer = OccupancyVisualizer(config=config, color_map=color_map, **kwargs)

        # Load external camera parameters if provided
        if view_json_path:
            try:
                visualizer.load_external_camera_parameters(view_json_path)
                logger.info(f"Pre-loaded camera parameters from {view_json_path}")
            except Exception as e:
                logger.warning(f"Could not load external camera parameters: {e}")

        return visualizer

    @staticmethod
    def create_batch_with_camera(output_dir: Union[str, Path],
                                color_map_name: str = 'occ3d',
                                view_json_path: Union[str, Path] = None,
                                **kwargs) -> Tuple[OccupancyVisualizer, 'BatchProcessor']:
        """
        Create batch processor with consistent external camera parameters.

        This method creates a batch processor that uses external camera parameters
        for all visualizations, ensuring consistent viewing angles across entire
        datasets or sequences.

        Args:
            output_dir: Directory for batch processing output
            color_map_name: Name of color map to use ('occ3d', 'openocc', 'default')
            view_json_path: Path to external view.json file with camera parameters
            **kwargs: Additional arguments passed to OccupancyVisualizer

        Returns:
            Tuple of (visualizer, batch_processor) with pre-loaded camera parameters

        Example:
            # Create batch processor with consistent camera angle
            visualizer, processor = VisualizerFactory.create_batch_with_camera(
                output_dir='batch_output',
                color_map_name='occ3d',
                view_json_path='my_camera_view.json'
            )

            # Process entire 4D sequence with consistent camera angle
            processor.process_4d_sequence(data_4d, scene_name='sequence1')
        """
        visualizer = VisualizerFactory.create_with_external_camera(
            color_map_name=color_map_name,
            view_json_path=view_json_path,
            **kwargs
        )

        processor = BatchProcessor(visualizer, output_dir)
        return visualizer, processor

    @staticmethod
    def create_batch_visualizer(output_dir: Union[str, Path],
                               color_map_name: str = 'occ3d',
                               **kwargs) -> Tuple[OccupancyVisualizer, BatchProcessor]:
        """
        Create visualizer with batch processor.

        Args:
            output_dir: Output directory for batch processing
            color_map_name: Name of color map to use ('occ3d', 'openocc', 'default')
            **kwargs: Additional arguments passed to OccupancyVisualizer
        """
        config = VisualizationConfig()
        color_map = get_color_map_by_name(color_map_name)
        visualizer = OccupancyVisualizer(config=config, color_map=color_map, **kwargs)
        processor = BatchProcessor(visualizer, output_dir)
        return visualizer, processor

    @staticmethod
    def create_nuscenes_processor(output_dir: Union[str, Path],
                                 color_map_name: str = 'occ3d',
                                 view_json_path: Optional[Union[str, Path]] = None,
                                 **kwargs) -> Tuple[OccupancyVisualizer, 'BatchProcessor']:
        """
        Create batch processor specifically configured for NuScenes data.

        This factory method creates a visualizer and batch processor optimized
        for processing NuScenes occupancy sequences with proper settings for
        the dataset characteristics and standard workflows.

        Args:
            output_dir: Directory for NuScenes visualization output
            color_map_name: Color map optimized for NuScenes ('occ3d' recommended)
            view_json_path: Optional path to camera parameters for consistent views
            **kwargs: Additional arguments passed to OccupancyVisualizer

        Returns:
            Tuple of (visualizer, batch_processor) configured for NuScenes

        Example:
            # Create NuScenes processor with consistent camera
            visualizer, processor = VisualizerFactory.create_nuscenes_processor(
                output_dir='nuscenes_results',
                color_map_name='occ3d',
                view_json_path='nuscenes_camera_view.json'
            )

            # Process NuScenes sequences
            results = processor.process_nuscenes_sequence(
                infos_path='nuscenes_infos_val.pkl',
                data_version='v1.0-mini',
                data_path='data/nuscenes',
                vis_scenes=['scene-0061', 'scene-0103']
            )
        """
        # NuScenes-optimized configuration
        config = VisualizationConfig(
            voxel_size=(0.4, 0.4, 0.4),  # Standard NuScenes voxel size
            range_vals=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),  # NuScenes range
            ignore_labels=[0, 17],  # Background and undefined classes
            point_size=4,
            show_coordinate_frame=True,
            frame_size=1.0,
            wait_time=0.5  # Good for sequence viewing
        )

        # Use OCC3D colors by default for NuScenes (18 classes)
        color_map = get_color_map_by_name(color_map_name)

        visualizer = OccupancyVisualizer(config=config, color_map=color_map, **kwargs)

        # Load external camera parameters if provided
        if view_json_path:
            try:
                visualizer.load_external_camera_parameters(view_json_path)
                logger.info(f"Pre-loaded NuScenes camera parameters from {view_json_path}")
            except Exception as e:
                logger.warning(f"Could not load camera parameters: {e}")

        processor = BatchProcessor(visualizer, output_dir)
        return visualizer, processor

    @staticmethod
    def create_nuscenes_prediction_processor(output_dir: Union[str, Path],
                                           color_map_name: str = 'occ3d',
                                           view_json_path: Optional[Union[str, Path]] = None,
                                           comparison_mode: str = 'side_by_side',
                                           **kwargs) -> Tuple[OccupancyVisualizer, 'BatchProcessor']:
        """
        Create batch processor specifically configured for NuScenes prediction comparison.

        This factory method creates a visualizer and batch processor optimized for
        comparing NuScenes ground truth occupancy data with prediction results.
        Supports multiple comparison modes including side-by-side, overlay, and difference views.

        Args:
            output_dir: Directory for prediction comparison output
            color_map_name: Color map optimized for NuScenes ('occ3d' recommended)
            view_json_path: Optional path to camera parameters for consistent views
            comparison_mode: Comparison visualization mode ('side_by_side', 'separate', 'overlay', 'difference')
            **kwargs: Additional arguments passed to OccupancyVisualizer

        Returns:
            Tuple of (visualizer, batch_processor) configured for NuScenes predictions

        Example:
            # Create prediction comparison processor
            visualizer, processor = VisualizerFactory.create_nuscenes_prediction_processor(
                output_dir='prediction_comparison',
                comparison_mode='side_by_side',
                view_json_path='consistent_view.json'
            )

            # Process GT vs predictions
            results = processor.process_nuscenes_predictions(
                infos_path='nuscenes_infos_val.pkl',
                pred_path='pred',
                vis_scenes=['scene-0061'],
                comparison_mode='side_by_side'
            )
        """
        # NuScenes-optimized configuration for predictions
        config = VisualizationConfig(
            voxel_size=(0.4, 0.4, 0.4),  # Standard NuScenes voxel size
            range_vals=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),  # NuScenes range
            ignore_labels=[0, 17],  # Background and undefined classes
            point_size=4,
            show_coordinate_frame=True,
            frame_size=1.0,
            wait_time=0.5,  # Good for prediction comparison viewing
            window_width=1200,  # Wider window for side-by-side comparisons
            window_height=800
        )

        # Use OCC3D colors by default for NuScenes predictions (18 classes)
        color_map = get_color_map_by_name(color_map_name)

        visualizer = OccupancyVisualizer(config=config, color_map=color_map, **kwargs)

        # Load external camera parameters if provided
        if view_json_path:
            try:
                visualizer.load_external_camera_parameters(view_json_path)
                logger.info(f"Pre-loaded camera parameters for prediction comparison from {view_json_path}")
            except Exception as e:
                logger.warning(f"Could not load camera parameters: {e}")

        processor = BatchProcessor(visualizer, output_dir)

        # Store comparison mode for reference
        processor._comparison_mode = comparison_mode

        logger.info(f"Created NuScenes prediction processor with comparison mode: {comparison_mode}")
        return visualizer, processor

    @staticmethod
    def create_bev_processor(output_dir: Union[str, Path],
                           dataset_type: str = 'nuscenes',
                           color_map_name: Optional[str] = None,
                           view_json_path: Optional[Union[str, Path]] = None,
                           **kwargs) -> Tuple[OccupancyVisualizer, 'BatchProcessor']:
        """
        Create batch processor specifically configured for BEV (Bird's Eye View) visualization.

        This factory method creates a visualizer and batch processor optimized for
        processing occupancy data in BEV format. BEV visualization projects 3D occupancy
        grids to 2D top-down views by finding the topmost non-free voxel for each location.

        Args:
            output_dir: Directory for BEV visualization output
            dataset_type: Dataset type for BEV optimization ('nuscenes', 'openocc')
            color_map_name: Color map for BEV visualization (auto-selected if None)
            view_json_path: Optional path to camera parameters (not used for BEV but for consistency)
            **kwargs: Additional arguments passed to OccupancyVisualizer

        Returns:
            Tuple of (visualizer, batch_processor) configured for BEV processing

        Example:
            # Create BEV processor for NuScenes data
            visualizer, processor = VisualizerFactory.create_bev_processor(
                output_dir='bev_results',
                dataset_type='nuscenes'
            )

            # Process 4D sequence in BEV format
            bev_images = processor.process_bev_sequence(
                data_path='4D_occupancy_data.npz',
                create_video=True
            )
        """
        # BEV-optimized configuration
        config = VisualizationConfig(
            voxel_size=(0.4, 0.4, 0.4),  # Standard for most datasets
            range_vals=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),  # Wide range for BEV
            ignore_labels=[BEVProcessor.get_dataset_free_class(dataset_type)],
            point_size=4,
            show_coordinate_frame=False,  # Not needed for BEV
            wait_time=0.5  # Good for BEV sequence viewing
        )

        # Use dataset-appropriate color map
        if color_map_name is None:
            color_map_name = 'occ3d' if dataset_type.lower() == 'nuscenes' else 'openocc'

        color_map = get_color_map_by_name(color_map_name)

        visualizer = OccupancyVisualizer(config=config, color_map=color_map, **kwargs)

        # Load external camera parameters if provided (for consistency)
        if view_json_path:
            try:
                visualizer.load_external_camera_parameters(view_json_path)
                logger.info(f"Pre-loaded camera parameters for BEV processor from {view_json_path}")
            except Exception as e:
                logger.warning(f"Could not load camera parameters: {e}")

        processor = BatchProcessor(visualizer, output_dir)

        logger.info(f"Created BEV processor for {dataset_type} dataset")
        return visualizer, processor

    @staticmethod
    def create_nuscenes_bev_processor(output_dir: Union[str, Path],
                                    color_map_name: str = 'occ3d',
                                    view_json_path: Optional[Union[str, Path]] = None,
                                    **kwargs) -> Tuple[OccupancyVisualizer, 'BatchProcessor']:
        """
        Create batch processor specifically configured for NuScenes BEV visualization.

        This factory method creates a visualizer and batch processor optimized for
        processing NuScenes occupancy data in BEV format with dataset-specific settings.

        Args:
            output_dir: Directory for NuScenes BEV output
            color_map_name: Color map optimized for NuScenes BEV ('occ3d' recommended)
            view_json_path: Optional path to camera parameters for consistency
            **kwargs: Additional arguments passed to OccupancyVisualizer

        Returns:
            Tuple of (visualizer, batch_processor) configured for NuScenes BEV

        Example:
            # Create NuScenes BEV processor
            visualizer, processor = VisualizerFactory.create_nuscenes_bev_processor(
                output_dir='nuscenes_bev_results'
            )

            # Process NuScenes sequences in BEV format
            results = processor.process_nuscenes_bev_sequence(
                infos_path='nuscenes_infos_val.pkl',
                vis_scenes=['scene-0061', 'scene-0103']
            )
        """
        # NuScenes BEV-optimized configuration
        config = VisualizationConfig(
            voxel_size=(0.4, 0.4, 0.4),  # Standard NuScenes voxel size
            range_vals=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),  # NuScenes range
            ignore_labels=[17],  # NuScenes free class
            point_size=4,
            show_coordinate_frame=False,  # Not needed for BEV
            frame_size=1.0,
            wait_time=0.5  # Good for BEV sequence viewing
        )

        # Use OCC3D colors by default for NuScenes BEV (18 classes)
        color_map = get_color_map_by_name(color_map_name)

        visualizer = OccupancyVisualizer(config=config, color_map=color_map, **kwargs)

        # Load external camera parameters if provided
        if view_json_path:
            try:
                visualizer.load_external_camera_parameters(view_json_path)
                logger.info(f"Pre-loaded camera parameters for NuScenes BEV from {view_json_path}")
            except Exception as e:
                logger.warning(f"Could not load camera parameters: {e}")

        processor = BatchProcessor(visualizer, output_dir)

        logger.info("Created NuScenes BEV processor with OCC3D color mapping")
        return visualizer, processor

    @staticmethod
    def create_bev_prediction_processor(output_dir: Union[str, Path],
                                      dataset_type: str = 'nuscenes',
                                      comparison_mode: str = 'side_by_side',
                                      color_map_name: Optional[str] = None,
                                      view_json_path: Optional[Union[str, Path]] = None,
                                      **kwargs) -> Tuple[OccupancyVisualizer, 'BatchProcessor']:
        """
        Create batch processor for BEV prediction vs ground truth comparison.

        This factory method creates a visualizer and batch processor optimized for
        comparing occupancy predictions with ground truth data in BEV format.

        Args:
            output_dir: Directory for BEV prediction comparison output
            dataset_type: Dataset type for BEV optimization ('nuscenes', 'openocc')
            comparison_mode: BEV comparison mode ('side_by_side', 'overlay', 'difference')
            color_map_name: Color map for BEV visualization (auto-selected if None)
            view_json_path: Optional path to camera parameters for consistency
            **kwargs: Additional arguments passed to OccupancyVisualizer

        Returns:
            Tuple of (visualizer, batch_processor) configured for BEV prediction comparison

        Example:
            # Create BEV prediction comparison processor
            visualizer, processor = VisualizerFactory.create_bev_prediction_processor(
                output_dir='bev_prediction_comparison',
                dataset_type='nuscenes',
                comparison_mode='side_by_side'
            )

            # Process predictions vs GT in BEV format
            results = processor.process_bev_predictions(
                infos_path='nuscenes_infos_val.pkl',
                pred_path='pred',
                comparison_mode='side_by_side'
            )
        """
        # BEV prediction comparison configuration
        config = VisualizationConfig(
            voxel_size=(0.4, 0.4, 0.4),  # Standard for most datasets
            range_vals=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),  # Wide range for BEV
            ignore_labels=[BEVProcessor.get_dataset_free_class(dataset_type)],
            point_size=4,
            show_coordinate_frame=False,  # Not needed for BEV
            wait_time=0.5,  # Good for BEV comparison viewing
            window_width=1200,  # Wider for side-by-side comparisons
            window_height=800
        )

        # Use dataset-appropriate color map
        if color_map_name is None:
            color_map_name = 'occ3d' if dataset_type.lower() == 'nuscenes' else 'openocc'

        color_map = get_color_map_by_name(color_map_name)

        visualizer = OccupancyVisualizer(config=config, color_map=color_map, **kwargs)

        # Load external camera parameters if provided
        if view_json_path:
            try:
                visualizer.load_external_camera_parameters(view_json_path)
                logger.info(f"Pre-loaded camera parameters for BEV prediction comparison from {view_json_path}")
            except Exception as e:
                logger.warning(f"Could not load camera parameters: {e}")

        processor = BatchProcessor(visualizer, output_dir)

        # Store comparison mode for reference
        processor._bev_comparison_mode = comparison_mode

        logger.info(f"Created BEV prediction processor: dataset={dataset_type}, mode={comparison_mode}")
        return visualizer, processor

    @staticmethod
    def create_custom_visualizer(color_map_name: str, **kwargs) -> OccupancyVisualizer:
        """
        Create visualizer with specified color map.

        Args:
            color_map_name: Name of color map to use ('occ3d', 'openocc', 'default')
            **kwargs: Additional arguments passed to OccupancyVisualizer
        """
        config = VisualizationConfig()
        color_map = get_color_map_by_name(color_map_name)
        return OccupancyVisualizer(config=config, color_map=color_map, **kwargs)


# Utility functions for backward compatibility and convenience
def create_default_color_map(num_classes: int) -> np.ndarray:
    """Create a default color map for semantic classes."""
    colors = []
    for i in range(num_classes):
        hue = i / max(1, num_classes - 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append([int(c * 255) for c in rgb])
    return np.array(colors)


def get_occ3d_color_map() -> np.ndarray:
    """
    Get the OCC3D standard color map for occupancy visualization.

    Returns:
        Color map array with shape (18, 3) in BGR format for Open3D compatibility

    Classes:
        0: others (black)
        1: barrier (orange)
        2: bicycle (pink)
        3: bus (yellow)
        4: car (blue)
        5: construction_vehicle (cyan)
        6: motorcycle (dark orange)
        7: pedestrian (red)
        8: traffic_cone (light yellow)
        9: trailer (brown)
        10: truck (purple)
        11: driveable_surface (dark pink)
        12: other_flat (dark red)
        13: sidewalk (dark purple)
        14: terrain (light green)
        15: manmade (white)
        16: vegetation (green)
        17: free (white)
    """
    occ3d_colors_map = np.array([
        [0, 0, 0],          # others               black
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
        [75, 0, 75],        # sidewalk             dark purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255],    # free                 white
    ])
    # Convert RGB to BGR for Open3D compatibility
    return occ3d_colors_map[:, ::-1]


def get_openocc_color_map() -> np.ndarray:
    """
    Get the OpenOCC standard color map for occupancy visualization.

    Returns:
        Color map array with shape (17, 3) in BGR format for Open3D compatibility

    Classes:
        0: car (blue)
        1: truck (purple)
        2: trailer (brown)
        3: bus (yellow)
        4: construction_vehicle (cyan)
        5: bicycle (pink)
        6: motorcycle (dark orange)
        7: pedestrian (red)
        8: traffic_cone (light yellow)
        9: barrier (orange)
        10: driveable_surface (dark pink)
        11: other_flat (dark red)
        12: sidewalk (dark purple)
        13: terrain (light green)
        14: manmade (white)
        15: vegetation (green)
        16: free (white)
    """
    openocc_colors_map = np.array([
        [0, 150, 245],      # car                  blue
        [160, 32, 240],     # truck                purple
        [135, 60, 0],       # trailer              brown
        [255, 255, 0],      # bus                  yellow
        [0, 255, 255],      # construction_vehicle cyan
        [255, 192, 203],    # bicycle              pink
        [255, 127, 0],      # motorcycle           dark orange
        [255, 0, 0],        # pedestrian           red
        [255, 240, 150],    # traffic_cone         light yellow
        [255, 120, 50],     # barrier              orange
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dark purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255],    # free                 white
    ])
    # Already in RGB format, convert to BGR for Open3D compatibility
    return openocc_colors_map[:, ::-1]


def get_color_map_by_name(name: str) -> np.ndarray:
    """
    Get color map by name.

    Args:
        name: Color map name ('occ3d', 'openocc', or 'default')

    Returns:
        Color map array in BGR format for Open3D compatibility

    Raises:
        ValueError: If color map name is not recognized
    """
    name = name.lower()
    if name in ['occ3d', 'occ_3d']:
        return get_occ3d_color_map()
    elif name in ['openocc', 'open_occ']:
        return get_openocc_color_map()
    elif name == 'default':
        return create_default_color_map(18)  # Default with 18 classes
    else:
        raise ValueError(f"Unknown color map name: {name}. "
                        f"Available options: 'occ3d', 'openocc', 'default'")


def get_class_names_occ3d() -> List[str]:
    """Get class names for OCC3D dataset."""
    return [
        'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation', 'free'
    ]


def get_class_names_openocc() -> List[str]:
    """Get class names for OpenOCC dataset."""
    return [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation', 'free'
    ]


def load_occupancy_data(file_path: Union[str, Path]) -> np.ndarray:
    """Load occupancy data from various file formats."""
    file_path = Path(file_path)

    if file_path.suffix == '.npz':
        data = np.load(file_path)
        # Try common key names
        for key in ['semantics', 'occupancy', 'labels', 'data']:
            if key in data:
                return data[key]
        # If no known key, return first array
        return data[list(data.keys())[0]]

    elif file_path.suffix == '.npy':
        return np.load(file_path)

    else:
        raise OccupancyVisualizerError(f"Unsupported file format: {file_path.suffix}")


# Example usage and demo functions
def demo_basic_visualization():
    """Demonstrate basic occupancy visualization with OCC3D colors."""
    # Create sample data with OCC3D class range
    occupancy = np.random.randint(0, 18, size=(100, 100, 20))

    # Create visualizer with OCC3D color map
    visualizer = VisualizerFactory.create_occ3d_visualizer()

    try:
        visualizer.visualize_occupancy(occupancy)
    finally:
        visualizer.cleanup()


def demo_openocc_visualization():
    """Demonstrate OpenOCC visualization with standard colors."""
    # Create sample data with OpenOCC class range
    occupancy = np.random.randint(0, 17, size=(80, 80, 16))

    # Create visualizer with OpenOCC color map
    visualizer = VisualizerFactory.create_openocc_visualizer()

    try:
        visualizer.visualize_occupancy(
            occupancy_data=occupancy,
            save_path="openocc_demo.png"
        )
        logger.info("OpenOCC demo visualization saved")
    finally:
        visualizer.cleanup()


def demo_color_map_comparison():
    """Demonstrate different color maps side by side."""
    # Create sample data
    occupancy = np.random.randint(1, 11, size=(60, 60, 12))

    # Demo different color maps
    color_maps = ['occ3d', 'openocc', 'default']

    for color_map_name in color_maps:
        logger.info(f"Demonstrating {color_map_name} color map")

        visualizer = VisualizerFactory.create_custom_visualizer(
            color_map_name=color_map_name
        )

        try:
            visualizer.visualize_occupancy(
                occupancy_data=occupancy,
                save_path=f"demo_{color_map_name}_colors.png"
            )
            logger.info(f"Saved {color_map_name} color map demo")
        finally:
            visualizer.cleanup()


def demo_flow_visualization():
    """Demonstrate flow visualization."""
    # Create sample data
    occupancy = np.random.randint(0, 5, size=(50, 50, 10))
    flow = np.random.randn(50, 50, 10, 2) * 0.5

    # Create flow visualizer
    visualizer = VisualizerFactory.create_flow_visualizer()

    try:
        visualizer.visualize_occupancy(
            occupancy_data=occupancy,
            flow_data=flow,
            mode=VisualizationMode.FLOW
        )
    finally:
        visualizer.cleanup()


def demo_batch_processing():
    """Demonstrate batch processing of 4D data with OCC3D colors."""
    # Create sample 4D data with OCC3D class range
    data_4d = np.random.randint(0, 18, size=(10, 60, 60, 12))  # 10 frames

    # Create batch processor with OCC3D color map
    visualizer, processor = VisualizerFactory.create_batch_visualizer(
        "demo_output",
        color_map_name="occ3d"
    )

    try:
        # Process sequence
        image_paths = processor.process_4d_sequence(
            data_4d=data_4d,
            scene_name="occ3d_demo_sequence",
            create_video=True,
            video_fps=2
        )
        print(f"Generated {len(image_paths)} images with OCC3D colors")

    finally:
        visualizer.cleanup()


def demo_class_legend():
    """Demonstrate how to print class information and colors."""
    print("\n=== OCC3D Classes ===")
    occ3d_classes = get_class_names_occ3d()
    occ3d_colors = get_occ3d_color_map()

    for i, (class_name, color) in enumerate(zip(occ3d_classes, occ3d_colors)):
        # Convert BGR back to RGB for display
        rgb_color = color[::-1]
        print(f"{i:2d}: {class_name:20s} RGB{tuple(rgb_color)}")

    print("\n=== OpenOCC Classes ===")
    openocc_classes = get_class_names_openocc()
    openocc_colors = get_openocc_color_map()

    for i, (class_name, color) in enumerate(zip(openocc_classes, openocc_colors)):
        # Convert BGR back to RGB for display
        rgb_color = color[::-1]
        print(f"{i:2d}: {class_name:20s} RGB{tuple(rgb_color)}")


if __name__ == "__main__":
    # Print class information
    demo_class_legend()

    # Run demos
    print("\nRunning OCC3D visualization demo...")
    demo_basic_visualization()

    print("Running OpenOCC visualization demo...")
    demo_openocc_visualization()

    print("Running color map comparison demo...")
    demo_color_map_comparison()

    print("Running flow visualization demo...")
    demo_flow_visualization()

    print("Running batch processing demo...")
    demo_batch_processing()