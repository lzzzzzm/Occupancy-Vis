#!/usr/bin/env python3
"""
Quick Start Guide for Occupancy Visualization

This script demonstrates the simplest way to visualize your occupancy data.
Run this script after installing dependencies: pip install numpy open3d opencv-python
"""

import numpy as np
from occupancy_visualizer import VisualizerFactory, VisualizationMode

def quick_demo():
    occupancy_data_occ3d = np.load('demo_data/3D_occupancy_data.npz')['semantics']        # (H, W, D), values in [0, num_classes-1]
    occupancy_data_openocc = np.load('demo_data/3D_occupancy_flow_data.npz')['semantics']
    occupancy_flow_data_openocc = np.load('demo_data/3D_occupancy_flow_data.npz')['flow']  # (H, W, D, 2), float32
    occupancy_4d_data = np.load('demo_data/4D_occupancy_data.npz')['semantics']  # (T, H, W, D), values in [0, num_classes-1]

    print("🚀 Quick Occupancy Visualization Demo")
    print("1. Visualizing 3D Occupancy Data with Occ3D Color Map")

    visualizer = VisualizerFactory.create_occ3d_visualizer()
    success = visualizer.visualize_occupancy(
        occupancy_data=occupancy_data_occ3d,
        save_path="quick_demo_3d.png",
    )
    visualizer.cleanup()
    print("Visualization success:", success)


    print("2. Visualizing flow data...")
    flow_visualizer = VisualizerFactory.create_openocc_visualizer()
    success = flow_visualizer.visualize_occupancy(
        occupancy_data=occupancy_data_openocc,
        flow_data=occupancy_flow_data_openocc,
        mode=VisualizationMode.FLOW,
        save_path="quick_demo_flow.png"
    )
    flow_visualizer.cleanup()
    print("Visualization success:", success)

    print("3. Visualizing first frame of 4D sequence...")
    # Create batch processor
    visualizer, processor = VisualizerFactory.create_batch_visualizer("output")
    # Process sequence and create video
    image_paths = processor.process_4d_sequence(
        data_4d=occupancy_4d_data,
        scene_name="my_sequence",
        create_video=True,
        video_fps=5,
    )
    print(f"Generated {len(image_paths)} images and video")
    print("Images saved to:", image_paths)
    print("Video saved to: output/my_sequence_video.avi")
    print("Video creation success:", len(image_paths) > 0)
    visualizer.cleanup()


    print("4. Visualizing Nuscenes 4D sequence...")
    # Create batch processor
    visualizer, processor = VisualizerFactory.create_batch_visualizer("output")
    results = processor.process_nuscenes_sequence(
        infos_path="demo_data/world-nuscenes_mini_infos_val.pkl",
        data_version="v1.0-mini",
        data_path="data/nuscenes",
        vis_scenes=["scene-0103"],
        wait_time=0.5,
        maintain_camera=True
    )
    print("Results:", results)
    visualizer.cleanup()

    print("5. Visualizing pred Nuscenes 4D sequence...")
    # Create batch processor
    visualizer, processor = VisualizerFactory.create_batch_visualizer("output")
    results = processor.process_nuscenes_predictions(
        infos_path="demo_data/world-nuscenes_mini_infos_val.pkl",
        pred_path='demo_data',
        data_version="v1.0-mini",
        data_path="data/nuscenes",
        vis_scenes=["scene-0103"],
        wait_time=0.5,
        maintain_camera=True,
        comparison_mode='separate'
    )
    visualizer.cleanup()

    print("6. Visualizing single BEV...")
    visualizer, processor = VisualizerFactory.create_bev_processor(
        output_dir="output",
        dataset_type='nuscenes'
    )
    visualizer.visualize_bev(
        occupancy_data=occupancy_data_occ3d,
        save_path="bev_view.png",
        dataset_type='nuscenes'
    )
    visualizer.cleanup()

    print("7. Visualizing 4D BEV...")
    visualizer, processor = VisualizerFactory.create_bev_processor(
        output_dir="output",
        dataset_type='nuscenes'
    )
    bev_images = processor.process_bev_sequence(
        occupancy_4d=occupancy_4d_data,
        create_video=True
    )
    visualizer.cleanup()


if __name__ == '__main__':
    quick_demo()