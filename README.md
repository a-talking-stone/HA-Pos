# HA-Pos

### Hierarchical Prompt-Guided Adaptive Detection for Cross-view Visual Positioning System
![architecture](architecture.jpg)

## Abstract

With the rapid proliferation of Location-Based Services, achieving high-precision self-positioning on consumer-grade mobile devices—such as smartphones and civil drones—remains a critical challenge, particularly in GPS-denied or multipath-prone urban environments. This paper proposes HA-Pos, a novel hierarchical adaptive prompting mechanism enhancing the Cross-view Visual Positioning System for consumer electronics. The proposed method enables target specification via a user-defined click on a query image captured by a consumer terminal, subsequently locating that object within corresponding satellite reference imagery. Unlike traditional methods struggling with cross-view geometric distortions, HA-Pos incorporates a Hierarchical Prompt Query Encoder. This encoder provides precise spatial guidance across various depth stages, significantly bolstering the ability to distinguish target objects from distractors. Building upon this, a Geometric Adaptive Decoupled Head is designed to improve geometric adaptability and positioning accuracy. The GAD-Head integrates deformable convolutions as a Deformation-Aware Module to effectively capture geometric variations while independently optimizing regression and classification tasks. Extensive experiments demonstrate that HA-Pos achieves state-of-the-art performance on the CVOGL benchmark dataset.

## Updates

[25 Mar. 2026]This paper is now available online via *IEEE Transactions on Consumer Electronics (TCE) [Early Access](https://ieeexplore.ieee.org/document/11450442)*.

## Dataset

CVOGL is a dataset that comprises ground-view images, drone aerial-view images, and satellite-view images.

Please download the following datasets:

- [CVOGL](https://drive.google.com/file/d/1WCwnK_rrU--ZOIQtmaKdR0TXcmtzU4cf/view?usp=sharing)

## Requirements

- torch>=2.0.0,<2.5.0
- torchvision>=0.15.0,<0.20.0
- numpy>=1.21.0,<2.0.0
- opencv-python>=4.5.0,<5.0.0
- albumentations>=1.3.0,<2.0.0
- timm>=0.9.0,<1.1.0
- einops>=0.6.0,<0.9.0
- thop>=0.1.1.post2209072231
- shapely>=2.0.0,<3.0.0
- matplotlib>=3.5.0,<4.0.0

## Train

1. Firstly, download CVOGL and rename it to 'data', i.e., 'data/CVOGL_DroneAerial' and 'data/CVOGL_SVI'.

2. Secondly, download the pretrained Yolov3 model and place it in the 'saved_models' directory, i.e.,

   ['./saved_models/yolov3.weights'](https://pan.baidu.com/s/1As2Z0e8hD2PLimplLdsReg)(code: iu7g).

3. Thirdly, execute 'scripts/train_all.sh', 'scripts/train_drone.sh', or 'scripts/train_ground.sh' to train the models.

```
sh scripts/train_all.sh
# sh scripts/train_drone.sh
# sh scripts/train_ground.sh
```

## Test

```
sh test_all.sh
# sh test_drone.sh
# sh test_ground.sh
```

## Related Projects

[DetGeo](https://ieeexplore.ieee.org/document/10226220) (The baseline method improved by HA-Pos, accepted to IEEE TGRS 2023)

## Citation

```
@ARTICLE{11450442,
  author={Zheng, Jiehao and Huang, Guoheng and Chen, Xiaoyong and Fang, Haoran and Zhao, Kaiqi and Yuan, Xiaochen and Ling, Bingo Wing-Kuen and Tsang, Kim-Fung and Chen, Guanli and Pun, Chi-Man},
  journal={IEEE Transactions on Consumer Electronics}, 
  title={HA-Pos: Hierarchical Prompt-Guided Adaptive Detection for Cross-view Visual Positioning System}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Visualization;Head;Location awareness;Accuracy;Robustness;Feature extraction;Consumer electronics;Adaptation models;Satellites;Transforms;Cross-view Visual Positioning;hierarchical prompting;geometry adaptability},
  doi={10.1109/TCE.2026.3676565}}
```
