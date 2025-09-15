# Occupancy Data Visualization Guide

A comprehensive guide for visualizing 3D occupancy grid data with semantic segmentation and optical flow using the OccupancyVisualizer.

## 📊 Data Format Specifications

### Supported Data Types

#### **1. 3D Occupancy Data**
- **Shape**: `(H, W, D)` - Height × Width × Depth
- **Data Type**: Integer (typically `int32` or `int64`)
- **Value Range**: `[0, num_classes-1]` where each value represents a semantic class
- **Description**: Static 3D occupancy grid with semantic labels

#### **2. 3D Occupancy Flow Data**
- **Shape**: `(H, W, D, 2)` - Height × Width × Depth × Flow_Components
- **Data Type**: `float32`
- **Value Range**: Float values representing flow vectors
- **Description**: Optical flow vectors for each occupied voxel `[vx, vy]`

#### **3. 4D Occupancy Sequence Data**
- **Shape**: `(T, H, W, D)` - Time × Height × Width × Depth  
- **Data Type**: Integer (typically `int32` or `int64`)
- **Value Range**: `[0, num_classes-1]` temporal sequence of occupancy grids
- **Description**: Time series of 3D occupancy grids for dynamic scenes

#### **4. Point Cloud Data** 
- **Points Shape**: `(n, 3)` - Number_of_Points × XYZ_Coordinates
- **Labels Shape**: `(n,)` - Optional semantic labels for each point
- **Colors Shape**: `(n, 3)` - Optional RGB colors for each point [0-255]
- **Data Types**: 
  - Points: `float32` or `float64`
  - Labels: `int32` or `int64`
  - Colors: `uint8` or `float32`
- **Description**: Direct point cloud data for LiDAR, 3D reconstructions, or sparse 3D data
- **Supported Formats**: `.npz`, `.npy`, `.txt`, `.pcd`, `.ply`

#### **5. Point Cloud Sequence Data** 
- **Points Sequence**: List of arrays, each with shape `(n_i, 3)`
- **Labels Sequence**: List of arrays, each with shape `(n_i,)` (optional)
- **Colors Sequence**: List of arrays, each with shape `(n_i, 3)` (optional)
- **Description**: Temporal sequences of point cloud data with varying point counts
- **Use Cases**: LiDAR sequences, moving sensor data, temporal 3D reconstruction


#### **6. Nuscenes-Occ3D Data Format**

- **Shape**: `(H, W, D)` - Height × Width × Depth
- **Data Type**: Integer (typically `int32` or `int64`)
- **Value Range**: `[0, num_classes-1]` where each value represents a semantic class
- **Description**: Make sure the pkl file of occupancy path is the same as [world-nuscenes_mini_infos_val.pkl](demo_data/world-nuscenes_mini_infos_val.pkl)

## Installation

### Requirements
- Python 3.8+
- NumPy
- Open3D (for 3D visualization)
- OpenCV (for image processing and BEV views)
- 
### Install Dependencies

```bash
pip install numpy open3d-python opencv-python
```

### Optional Dependencies
```bash
# For NuScenes dataset support
pip install nuscenes-devkit

# For enhanced video processing
pip install imageio[ffmpeg]
```

## 🚀 Quick Start

### Running the Demo
```bash
python quick_start.py
```

## 📖 Detailed Usage Guide

### 1. 3D Occupancy Visualization

#### Basic 3D Visualization
```python
import numpy as np
from occupancy_visualizer import VisualizerFactory

# Load 3D occupancy data
occupancy_data = np.load('demo_data/3D_occupancy_data.npz')['semantics']

# Create visualizer with OCC3D color scheme
visualizer = VisualizerFactory.create_occ3d_visualizer()

# Visualize with default settings
visualizer.visualize_occupancy(
    occupancy_data=occupancy_data,
    save_path="3d_occupancy.png"
)
visualizer.cleanup()
```

### 2. Flow Visualization

#### Visualizing Optical Flow
```python
from occupancy_visualizer import VisualizationMode, VisualizerFactory
import numpy as np

# Load occupancy and flow data
data = np.load('demo_data/3D_occupancy_flow_data.npz')
occupancy_data = data['semantics']  # Shape: (H, W, D)
flow_data = data['flow']           # Shape: (H, W, D, 2)

# Create flow visualizer
visualizer = VisualizerFactory.create_flow_visualizer()

# Visualize flow vectors
visualizer.visualize_occupancy(
    occupancy_data=occupancy_data,
    flow_data=flow_data,
    mode=VisualizationMode.FLOW,
    save_path="flow_visualization.png"
)

# Combined semantic + flow visualization
visualizer.visualize_occupancy(
    occupancy_data=occupancy_data,
    flow_data=flow_data,
    mode=VisualizationMode.COMBINED,
    save_path="combined_visualization.png"
)
visualizer.cleanup()
```

### 3. 4D Sequence Processing

#### Processing Temporal Sequences
```python
from occupancy_visualizer import VisualizationMode, VisualizerFactory
import numpy as np
# Load 4D sequence data
sequence_data = np.load('demo_data/4D_occupancy_data.npz')['semantics']  # Shape: (T, H, W, D)

# Create batch processor
visualizer, processor = VisualizerFactory.create_batch_visualizer("output_folder")

# Process entire sequence and create video
image_paths = processor.process_4d_sequence(
    data_4d=sequence_data,
    scene_name="my_sequence",
    create_video=True,
    video_fps=5,
)

print(f"Generated {len(image_paths)} frames")
print("Video saved as: output_folder/my_sequence_video.mp4")
visualizer.cleanup()
```

### 4. Bird's Eye View (BEV) Visualization

#### Creating BEV Images
```python
# Single BEV visualization
success = visualizer.visualize_bev(
    occupancy_data=occupancy_data,
    save_path="bev_view.png",
    height_range=(-1.0, 5.4),  # Z-axis range to include
    resolution=0.5,            # meters per pixel
    dataset_type='occ3d'
)

# BEV comparison (prediction vs ground truth)
visualizer.visualize_bev_comparison(
    pred_data=prediction_data,
    gt_data=ground_truth_data,
    save_path="bev_comparison.png",
    height_range=(-1.0, 5.4)
)
```

### 5. NuScenes Dataset Integration
```python
import pickle

# Load NuScenes info file
with open('demo_data/world-nuscenes_mini_infos_val.pkl', 'rb') as f:
    infos = pickle.load(f)

# Process specific scene
scene_token = "scene-0103"
processor.process_nuscenes_sequence(
    infos=infos,
    scene_token=scene_token,
    occ_path="demo_data/",
    create_video=True,
    video_fps=2
)
```

### 6. Custom Configuration

#### Advanced Visualization Settings
```python
from occupancy_visualizer import OccupancyVisualizer, VisualizationConfig

# Create custom configuration
config = VisualizationConfig(
    voxel_size=0.4,                    # Voxel size in meters
    range_vals=[-40, -40, -1, 40, 40, 5.4],  # [x_min, y_min, z_min, x_max, y_max, z_max]
    ignore_labels=[17],                # Labels to ignore (e.g., free space)
    show_ego_car=True,                # Show ego vehicle
    point_size=2.0,                   # Point cloud point size
    background_color=[0, 0, 0],       # Black background
)

# Create visualizer with custom config
visualizer = OccupancyVisualizer(config=config)
```

### 7. Adding 3D Models

#### Using Car Models
```python
# Visualize with custom car model
visualizer.visualize_occupancy(
    occupancy_data=occupancy_data,
    car_model_path="demo_data/3d_model.obj",  # Path to 3D model file
    save_path="with_car_model.png"
)
```

#### Custom Camera Views
```python
# Save and load camera views
visualizer.visualize_occupancy(
    occupancy_data=occupancy_data,
    view_json_path="camera_view.json",  # Will save current view
    save_path="custom_view.png"
)

# Reuse saved view
visualizer.visualize_occupancy(
    occupancy_data=another_data,
    view_json_path="camera_view.json",  # Will load saved view
    save_path="same_view_different_data.png"
)
```


## ⚙️ Configuration Options

### VisualizationConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voxel_size` | float | 0.4 | Size of each voxel in meters |
| `range_vals` | List[float] | [-40, -40, -1, 40, 40, 5.4] | Spatial range [x_min, y_min, z_min, x_max, y_max, z_max] |
| `ignore_labels` | List[int] | [17] | Class labels to ignore during visualization |
| `show_ego_car` | bool | True | Whether to display ego vehicle |
| `window_size` | Tuple[int, int] | (1024, 768) | Render window size (width, height) |
| `point_size` | float | 1.0 | Size of points in point cloud |
| `background_color` | List[float] | [1, 1, 1] | Background color [R, G, B] (0-1 range) |

### Visualization Modes

| Mode | Description | Required Data |
|------|-------------|---------------|
| `VisualizationMode.SEMANTIC` | Show semantic classes with colors | occupancy_data |
| `VisualizationMode.FLOW` | Show optical flow vectors | occupancy_data + flow_data |
| `VisualizationMode.COMBINED` | Show both semantic and flow | occupancy_data + flow_data |

### Factory Methods

#### Occupancy Visualizers
| Method | Description | Best For |
|--------|-------------|----------|
| `create_occ3d_visualizer()` | OCC3D dataset colors | Standard occupancy grids |
| `create_openocc_visualizer()` | OpenOCC dataset colors | OpenOCC format data |
| `create_flow_visualizer()` | Optimized for flow data | Motion visualization |

#### Point Cloud Visualizers
| Method | Description | Best For |
|--------|-------------|----------|
| `create_point_cloud_visualizer()` | General point cloud visualization | Any point cloud data |
| `create_lidar_visualizer()` | Optimized for LiDAR data | Automotive LiDAR scans |
| `create_point_cloud_batch_processor()` | Batch processing with video output | Point cloud sequences |

### Color Schemes

| Scheme | Description | Use Case |
|--------|-------------|----------|
| `occ3d` | OCC3D standard colors | General 3D occupancy |
| `openocc` | OpenOCC color scheme | OpenOCC dataset |
| `nuscenes` | NuScenes color mapping | NuScenes dataset |
| Custom | User-defined colors | Specific requirements |

### Video Export Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_fps` | int | 10 | Frames per second |
| `video_format` | str | 'mp4' | Video format ('mp4', 'avi', 'gif') |
| `video_quality` | int | 5 | Video quality (1-10, higher is better) |
| `create_preview` | bool | True | Generate preview images |

## 🐛 Troubleshooting

### Common Issues

1. **Camera parameters not loading**
   ```python
   # Check if file exists and is valid
   if visualizer.has_camera_parameters():
       print("✅ Camera parameters loaded")
   else:
       print("❌ No camera parameters available")
   ```

2. **4D sequence not auto-advancing**
   ```python
   # Ensure wait_time is set properly
   visualizer, processor = VisualizerFactory.create_4d_processor(
       wait_time=1.0  # Must be > 0 for auto-advance
   )
   ```

3. **NuScenes data not found**
   ```python
   # Verify file paths
   import os
   pkl_path = "world-nuscenes_mini_infos_val.pkl"
   nuscenes_data = "data/nuscenes"
   
   print(f"PKL exists: {os.path.exists(pkl_path)}")
   print(f"Data dir exists: {os.path.exists(nuscenes_data)}")
   ```

4. **Memory issues with large sequences**
   ```python
   # Process in smaller batches
   results = processor.process_4d_sequence(
       data_path="large_sequence.npz",
       max_frames=50,  # Limit frames per batch
       cleanup_between_frames=True
   )
   ```

5. **Prediction files not found**
   ```python
   # Verify prediction file structure
   import os
   pred_path = "pred/scene-0061/sample_001/labels.npz"
   gt_path = "data/nuscenes/gts/scene-0061/sample_001/labels.npz"
   
   print(f"Prediction exists: {os.path.exists(pred_path)}")
   print(f"GT exists: {os.path.exists(gt_path)}")
   
   # Check data compatibility
   if os.path.exists(pred_path) and os.path.exists(gt_path):
       import numpy as np
       pred_data = np.load(pred_path)['semantics']
       gt_data = np.load(gt_path)['semantics']
       print(f"Shape match: {pred_data.shape == gt_data.shape}")
   ```

6. **Comparison visualization issues**
   ```python
   # Ensure comparison mode is valid
   valid_modes = ['side_by_side', 'separate', 'overlay', 'difference']
   comparison_mode = 'side_by_side'  # Choose from valid modes
   
   # Check data label ranges
   unique_gt = np.unique(gt_data)
   unique_pred = np.unique(pred_data)
   print(f"GT labels: {unique_gt}")
   print(f"Pred labels: {unique_pred}")
   ```