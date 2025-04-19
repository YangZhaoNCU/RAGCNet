# RAGCNet
This is the official code repository for "Few sampling meshes-based 3D tooth segmentation via region-aware graph
convolutional network". {[ESWA](https://doi.org/10.1016/j.eswa.2024.124255)}

## Abstract
Precise segmentation of teeth from intraoral scanner images is crucial for computer-assisted orthodontic
treatment planning, yet current segmentation quality often falls below clinical standards due to intricate tooth
morphology and blurred gingival lines. Previous deep learning-based methods typically focus on localized tooth
information, emphasizing detailed relations between each tooth while disregarding the holistic information
of tooth models. Furthermore, unique geometric information such as the centroid position of teeth remains
underutilized. To address these issues, we propose a Region-Aware Graph Convolutional Network (RAGCNet)
for 3D tooth segmentation, which is capable of effectively handling both local fine-grained details and global
holistic feature with few sampling meshes. Specifically, considering the differences in intraoral scanning
accuracy, we sample central meshes using an improved Farthest Point Sampling (FPS) algorithm, and then
aggregate the information of neighbor meshes using the K-Nearest Neighbor (KNN) method. Meanwhile, a
specially designed Region-Aware Module (RAM) via attention mechanism is proposed for feature extraction
and fusion. Additionally, we propose a novel Centroid Loss based on tooth centroid coordinates to impose
additional constraints on segmentation results. Evaluation on real datasets with 3D intraoral scanner-acquired
tooth mesh models demonstrates that RAGCNet outperforms other SOTA methods in 3D tooth segmentation.

## 0. Main Environments

- python==3.7.13
- torch==1.13.0+cu116 
- torch-geometric==2.3.1 
- torchaudio==0.13.0+cu116
- packaging==23.0
- timm==0.9.6
- sklearn==0.0.post1
- pointnet2-ops @ file:///RAGCNet/pointnet2_ops_lib
- KNN-CUDA @ file:///RAGCNet/knn_cuda



## 1. Prepare the dataset

### 3D-IOSSeg datasets
- The 3D-IOSSeg datasets can be found here {[3D-IOSSeg](https://reurl.cc/0vjLXY)}

- After downloading the datasets, you are supposed to put them into './data/train/' and './data/test/', and the file format reference is as follows.

- './data/'
  - train
    - .ply
  - test
    - .ply



## 2. Train the RAGCNet
```bash
cd RAGCNet
python train.py  # Train and test RAGCNet on the 3D-IOSSeg dataset.
```

## 3. Obtain the outputs
- After trianing, you could obtain the results in './expirement/'

## 4. Citation

- If you find our work useful in your research, please cite:

Yang Zhao, et al. "Few sampling meshes-based 3D tooth segmentation via region-aware graph convolutional network." Expert Systems with Applications 252 (2024): 124255.