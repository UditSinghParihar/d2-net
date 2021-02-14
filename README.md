## D2-Net Training on Custom Dataset

1. Training D2Net:  
	1. `python train2.py --imgPairs '/home/cair/backup/d2-net/data/imagePairs.csv' --poses '/home/cair/backup/d2-net/data/poses2W.npy' --K '/home/cair/backup/d2-net/data/K.npy' --dataset_path '/home/cair/backup/d2-net' --plot`

	2. `python train2.py --imgPairs '/home/cair/backup/d2-net/data_gazebo/img_pairs.csv' --poses '/home/cair/backup/d2-net/data_gazebo/poses2W.npy' --K '/home/cair/backup/d2-net/data_gazebo/K.npy' --dataset_path '/home/cair/backup/d2-net/data_gazebo' --plot`  
	
	3. `python train2.py --imgPairs /home/dhagash/udit/d2-net/data_gazebo/img_pairs.csv --poses /home/dhagash/udit/d2-net/data_gazebo/poses2W.npy --K /home/dhagash/udit/d2-net/data_gazebo/K.npy --dataset_path /home/dhagash/udit/d2-net/data_gazebo --plot`  

	4. `python train2.py --dataset_path /scratch/udit/phototourism/brandenburg_gate/dense/images/ --plot`  
	5. `python train2.py --dataset_path /scratch/udit/robotcar/overcast/2014-06-26-09-24-58/mono_top/ --dataset_path2 /scratch/udit/robotcar/overcast/2014-06-26-09-24-58/stereo/centre_top/ --plot`  

2. Training would generate:  
	1. `train_vis/ checkpoints/ log.txt`

3. Input data format:  
	1. Left hand coordinate system (Z forward, Y Down, X Right).  
	2. `pose1`: World wrt to camera 1.  
	3. Depth file stores values in meters.  

4. Extracting and Matching D2Net features:  
	1. `python extractMatch.py '/home/cair/backup/deep_floor/correspond/qualitative/pair_2/homo1.jpg' '/home/cair/backup/deep_floor/correspond/qualitative/pair_2/homo2.jpg'`  
	2. `python extractMatchRot.py /scratch/dhagash/phototourism/brandenburg_gate/dense/images/41967863_5418850004.jpg`    

5. Warping function:  
	1. `python testWarpTorch.py`  

6. Downloading ScanNet:
	1. `wget -r -A .sens -I /ScanNet/v2/scans/scene000*,/ScanNet/v2/scans/scene001*,/ScanNet/v2/scans/scene002*,/ScanNet/v2/scans/scene003*,/ScanNet/v2/scans/scene004* -nH --cut-dirs=3 --no-parent --reject="index.html*" http://datasets.rrc.iiit.ac.in/ScanNet/v2/scans/`  

7. Testing dataloader:  
	1. `cp lib/datasetGrid.py ./ && python datasetGrid.py /scratch/dhagash/phototourism/brandenburg_gate/dense/images/`  

8. Tensorboard:
	1. `ssh -L localhost:16006:localhost:6006 `  

9. Debugging cuda error:  
	1. `CUDA_LAUNCH_BLOCKING=1 python [YOUR_PROGRAM]`  

10. Generating top view using rgb and depth on gazebo dataset:
	1. `python getTopView.py data_gazebo/data5/rgb/rgb000000.jpg data_gazebo/data5/depth/depth000000.npy`  
11. Conda environment:
	1. Base environment activate: `conda activate`  
	2. Conda environment deactivate: `conda deactivate`
	3. Conda environments: `/home/udit/softwares/py37/envs/`  
	4. Activating other conda envs: `conda activate planercnn`    
	5. Removing conda env: `conda env remove --name ENVIRONMENT`  

12. Generating top view using normal and bilinear interpolation:  
	1. `python getTopViewNormal.py data_gazebo_floor/data7/rgb/rgb000000.jpg data_gazebo_floor/data7/depth/depth000000.npy`  
	2. `python getTopNormalH.py '/home/cair/backup/d2-net/data_gazebo_floor/data7/rgb/rgb000000.jpg' '/home/cair/backup/d2-net/data_gazebo_floor/data7/depth/depth000000.npy'`  
	3. `python getTopRealSense.py '/home/cair/backup/d2-net/dataVO/data11/rgb/rgb000000.jpg' '/home/cair/backup/d2-net/dataVO/data11/depth/depth000000.png'`  
 
13. Undistorting and coloring raw robotcar images, using robotcar-dataset-sdk:  
	1. `python play_images.py ../../overcast/2014-06-26-09-24-58/stereo/centre/ --models_dir ../models/`  

14. Generating image pairs from front and rear camera:  
	1. `python getPairsOxford.py /scratch/udit/robotcar/overcast/2014-06-26-09-24-58/vo/vo.csv /scratch/udit/robotcar/overcast/2014-06-26-09-24-58/stereo/centre_rgb /scratch/udit/robotcar/overcast/2014-06-26-09-24-58/mono_rear_rgb`    
	2. `python gpsPrior.py /scratch/udit/robotcar/overcast/2014-06-26-09-24-58/vo/vo.csv '/scratch/udit/robotcar/overcast/2014-06-26-09-24-58/vpr/front_top' '/scratch/udit/robotcar/overcast/2014-06-26-09-24-58/vpr/rear_top'`  

15. Viewing image pairs:  
	1. `python viewPairs.py imagePairsOxford.csv`  
  
16. Accessing tensorBoard on local machine while training on server:  
	1. `ssh -L localhost:16006:localhost:6006 udit@blue.iiit.ac.in`  
	2. `tensorboard --logdir runs/`  # On Server
	3. `http://localhost:16006/`   # On Local machine

17. VPR performance benchmark:  
	1. `python vprPairs.py dataGenerate/probPairs.csv '/scratch/udit/robotcar/overcast/2014-06-26-09-24-58/vpr/front_top' '/scratch/udit/robotcar/overcast/2014-06-26-09-24-58/vpr/rear_top'`    
	2. `python saveFeatures.py --dir '/scratch/udit/robotcar/overcast/2014-06-26-09-24-58/vpr/front_top'`  
	3. `python testMatcher.py '/scratch/udit/robotcar/overcast/ipm/front/25.png' '/scratch/udit/robotcar/overcast/ipm/rear/25.png'`  
	4. `python evaluate/plotCorr.py dataGenerate/vprOutput.csv`  
	5. `python evaluate/evalVPR.py dataGenerate/probPairs.csv dataGenerate/vprOutput.csv`  
	6. python extractDir.py --dir1 '/scratch/udit/robotcar/overcast/ipm2/front_top' --dir2 '/scratch/udit/robotcar/overcast/ipm2/rear_top'

18. IPM generation:  
	1. `python savePairs.py '/home/udit/udit/d2-net/dataGenerate/imagePairsOxford.csv'`  
	2. `python saveIpm.py rootDir cameraCfg.json`	

19. ROS:  
	1. `#rosbag record -O test4 camera/aligned_depth_to_color/image_raw /camera/color/image_raw /camera/color/camera_info /tf_static /rtabmap/odom /tf`  
	2. `roslaunch rtabmap_ros rtabmap.launch rtabmap_args:="--delete_db_on_start" depth_topic:=/camera/aligned_depth_to_color/image_raw rgb_topic:=/camera/color/image_raw camera_info_topic:=/camera/color/camera_info`  
	3. `roslaunch realsense2_camera rs_camera.launch align_depth:=true enable_gyro:=true enable_accel:=true unite_imu_method:=linear_interpolation`  

20. Evaluation on custom:  
	1. `python getGTCorr.py '/home/cair/backup/d2-net/dataVO/data11/poses.txt' '/home/cair/backup/d2-net/dataVO/data11/rgb' '/home/cair/backup/d2-net/dataVO/data11/depth'`  
	2. `python getTopRealSense.py ../dataVO/data11/rgb/rgb001000.jpg ../dataVO/data11/depth/depth001000.png`  
	3. `python extractMatchTop.py dataVO/data11/rgb/rgb000100.jpg dataVO/data11/rgb/rgb001000.jpg quantitative/top100.npy quantitative/top1000.npy`  
	4. `python icpCorr.py '/home/cair/backup/d2-net/dataVO/data11/rgb/rgb000100.jpg' '/home/cair/backup/d2-net/dataVO/data11/depth/depth000100.png' '/home/cair/backup/d2-net/dataVO/data11/rgb/rgb001000.jpg' '/home/cair/backup/d2-net/dataVO/data11/depth/depth001000.png' src_pts.txt trg_pts.txt`  
	5. Input: `rgb1, depth1, rgb2, depth2, H1, H2`. Ouput: `predicted [R T]`:  
		1. `python extractMatchIcp.py dataVO/data1/rgb/rgb000100.jpg dataVO/data1/depth/depth000100.png dataVO/data1/rgb/rgb001000.jpg dataVO/data1/depth/depth001000.png quantitative/top100.npy quantitative/top1000.npy`  


# [D2-Net: A Trainable CNN for Joint Detection and Description of Local Features](https://github.com/mihaidusmanu/d2-net)

This repository contains the implementation of the following paper:

```text
"D2-Net: A Trainable CNN for Joint Detection and Description of Local Features".
M. Dusmanu, I. Rocco, T. Pajdla, M. Pollefeys, J. Sivic, A. Torii, and T. Sattler. CVPR 2019.
```

[Paper on arXiv](https://arxiv.org/abs/1905.03561), [Project page](https://dsmn.ml/publications/d2-net.html)
    
## Getting started

Python 3.6+ is recommended for running our code. [Conda](https://docs.conda.io/en/latest/) can be used to install the required packages:

```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install h5py imageio imagesize matplotlib numpy scipy tqdm
```

## Downloading the models

The off-the-shelf **Caffe VGG16** weights and their tuned counterpart can be downloaded by running:

```bash
mkdir models
wget https://dsmn.ml/files/d2-net/d2_ots.pth -O models/d2_ots.pth
wget https://dsmn.ml/files/d2-net/d2_tf.pth -O models/d2_tf.pth
wget https://dsmn.ml/files/d2-net/d2_tf_no_phototourism.pth -O models/d2_tf_no_phototourism.pth
```

**Update - 23 May 2019** We have added a new set of weights trained on MegaDepth without the PhotoTourism scenes (sagrada_familia - 0019, lincoln_memorial_statue - 0021, british_museum - 0024, london_bridge - 0025, us_capitol - 0078, mount_rushmore - 1589). Our initial results show similar performance. In order to use these weights at test time, you should add `--model_file models/d2_tf_no_phototourism.pth`.

## Feature extraction

`extract_features.py` can be used to extract D2 features for a given list of images. The singlescale features require less than 6GB of VRAM for 1200x1600 images. The `--multiscale` flag can be used to extract multiscale features - for this, we recommend at least 12GB of VRAM. 

The output format can be either [`npz`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) or `mat`. In either case, the feature files encapsulate three arrays: 

- `keypoints` [`N x 3`] array containing the positions of keypoints `x, y` and the scales `s`. The positions follow the COLMAP format, with the `X` axis pointing to the right and the `Y` axis to the bottom.
- `scores` [`N`] array containing the activations of keypoints (higher is better).
- `descriptors` [`N x 512`] array containing the L2 normalized descriptors.

```bash
python extract_features.py --image_list_file images.txt (--multiscale)
```

## Tuning on MegaDepth

The training pipeline provided here is a PyTorch implementation of the TensorFlow code that was used to train the model available to download above.

**Update - 05 June 2019** We have fixed a bug in the dataset preprocessing - retraining now yields similar results to the original TensorFlow implementation.

**Update - 07 August 2019** We have released an updated, more accurate version of the training dataset - training is more stable and significantly faster for equal performance.

### Downloading and preprocessing the MegaDepth dataset

For this part, [COLMAP](https://colmap.github.io/) should be installed. Please refer to the official website for installation instructions.

After downloading the entire [MegaDepth](http://www.cs.cornell.edu/projects/megadepth/) dataset (including SfM models), the first step is generating the undistorted reconstructions. This can be done by calling `undistort_reconstructions.py` as follows:

```bash
python undistort_reconstructions.py --colmap_path /path/to/colmap/executable --base_path /path/to/megadepth
```

Next, `preprocess_megadepth.sh` can be used to retrieve the camera parameters and compute the overlap between images for all scenes. 

```bash
bash preprocess_undistorted_megadepth.sh /path/to/megadepth /path/to/output/folder
```

In case you prefer downloading the undistorted reconstructions and aggregated scene information folder directly, you can find them [here - Google Drive](https://drive.google.com/open?id=1hxpOsqOZefdrba_BqnW490XpNX_LgXPB). You will still need to download the depth maps ("MegaDepth v1 Dataset") from the MegaDepth website.

### Training

After downloading and preprocessing MegaDepth, the training can be started right away:

```bash
python train.py --use_validation --dataset_path /path/to/megadepth --scene_info_path /path/to/preprocessing/output
```

## BibTeX

If you use this code in your project, please cite the following paper:

```bibtex
@InProceedings{Dusmanu2019CVPR,
    author = {Dusmanu, Mihai and Rocco, Ignacio and Pajdla, Tomas and Pollefeys, Marc and Sivic, Josef and Torii, Akihiko and Sattler, Torsten},
    title = {{D2-Net: A Trainable CNN for Joint Detection and Description of Local Features}},
    booktitle = {Proceedings of the 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2019},
}
```
