## Installation
After cloned, assuming your workspace directory is `equiv_align/`: `cd equiv_align`

### Docker 
1. Download docker-compose [binary](https://github.com/docker/compose/releases) file to `~/.docker/cli-plugins/docker-compose` 
2. Build docker image and container
```
cd equiv_align
cd docker
sh build_imageg.sh
sh build_container.sh
```
3. After building the container, you will obtain a contianer (with `docker ps -a`), named like `docker-slam-run-[number]`. If you want to enter the container shell from the host machine, run `docker start [container-name]` to start the container and `docker exec -u ${USER} -it [container-name] /bin/bash`

### pip
Assume you are inside the container: 
```
cd equiv_align
python3 -m venv equiv_align_env
source equiv_align_env/bin/activate
cat requirements.txt | xargs -n 1 pip install
```
From now on, we assume that we are operating inside the container and venv.

## Dataset
### Download
Dataset Link: [Aligned ModelNet40, from E2PN repo](https://drive.google.com/file/d/1xRoYjz2KCwkyIPf21E-WKIZkjLYabPgJ/view?usp=sharing) and [ETH3D](https://www.eth3d.net/slam_datasets)
### Setup the dataset path inside docker container
In docker container:
```
$ cd equiv_align
$ pwd data 
/home/[USER-NAME]/equiv_align/data
$ ls data
### data folder
data
├── eth3d
│   ├── test
│   └── training
│       ├── cables_1
│       │   ├── depth
│       │   └── rgb
│       ├── ...
└── modelnet
    ├── EvenAlignedModelNet40PC
    │   ├── airplane
    │   │   ├── test
    │   │   ├── testR
    │   │   └── train
    │   ├── ...
```

## Run training and testing
### Training
#### Modelnet
In docker container and venv:
1. largescale training for modelnet: `sh scripts/train_modelnet_se3_largescale.sh`
2. training only on the airplane class for modelnet: `sh scripts/train_modelnet_se3_airplane_only.sh`

#### ETH3D and TUM format
In docker container and venv:
1. largescale training for eth3d: `sh scripts/train_eth3d_se3_largescale.sh`

#### Tensorboard
Launch the tensorboard: `sh scripts/launch_tensorboard.sh`

### Pretrained weights
Weights url: [Dropbox Link](https://www.dropbox.com/scl/fo/ye8vjz0x9mfpbu8224o3q/AIXIL5DPFqzkvUYfLoqEHKM?rlkey=37ogl2mj4bn7z33v9zm0bhl0d&st=2dwezc8t&dl=0)
1. `modelnet`:  Put the [weight file](https://www.dropbox.com/scl/fo/ye8vjz0x9mfpbu8224o3q/APJlNdIh6bqrXGjQvWe6m_8/modelnet_full_90/checkpoints?rlkey=37ogl2mj4bn7z33v9zm0bhl0d&subfolder_nav_tracking=1&st=62dsjxb3&dl=0) into `log/modelnet_full_90/checkpoints/`
2. `eth3d`: Put the [weight file](https://www.dropbox.com/scl/fo/ye8vjz0x9mfpbu8224o3q/AEpnVjsW1xKowFKBjDsjEpI/eth3d/checkpoints?rlkey=37ogl2mj4bn7z33v9zm0bhl0d&subfolder_nav_tracking=1&st=nzcag6cm&dl=0) into `log/eth3d/checkpoints/`

### Testing
Note that when testing, batch_size = 1 for each gpu is recommended. In docker container and venv:
1. Test modelnet: `sh scripts/test_modelnet_se3.sh log/[pretrained-model-dir]/checkpoints/`
2. Test eth3d: `sh scripts/test_eth3d_se3_largescale.sh log/[pretrained-model-dir]/checkpoints/`



## Citation
If you find the repo useful, please consider citing the [paper](https://arxiv.org/abs/2407.20223) as:
```
@InProceedings{ZhangRay24equivalign,
author="Zhang, Ray
and Zhou, Zheming
and Sun, Min
and Ghasemalizadeh, Omid
and Kuo, Cheng-Hao
and Eustice, Ryan M.
and Ghaffari, Maani
and Sen, Arnie",
title="Correspondence-Free SE(3) Point Cloud Registration in RKHS via Unsupervised Equivariant Learning",
booktitle="Computer Vision -- ECCV 2024",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="68--86",
}
```

