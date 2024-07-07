# Occupancy-Vis

## Installation

Install the pytorch and mmcv based on your GPU and cuda version.
```
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
```
Install the required packages.
```
pip install -r requirements.txt
```

## Data Preparation

Download the nuScenes dataset from [nuScenes](https://www.nuscenes.org/). The dataset should be organized as follows:
```
Occupancy-Vis
    data
        nuscenes
            gts                 # CVPR2023, refer to:https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction
            openoccv2           # CVPR2024, refer to:https://github.com/OpenDriveLab/OccNet
            maps
            samples
            sweeps
            v1.0-trainval
    nuscenes_infos_train.pkl    # generate by mmdet3d-v1.0rc4, please refer to the official document
    nuscenes_infos_val.pkl      # generate by mmdet3d-v1.0rc4, please refer to the official document
```

Your prediction file follows the format of the ground truth file. The prediction file should be organized as follows:
```
predictions
    scene_token.npz
    ....
```
