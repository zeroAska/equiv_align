## Installation
Assume your workspace directory is `equiv_align/`

### Docker 
1. Download docker-compose [binary](https://github.com/docker/compose/releases) file to `~/.docker/cli-plugins/docker-compose` 
2. Build docker image and container
```
cd docker
sh build_imageg.sh
sh build_container.sh
```
3. After building the container, you will obtain a contianer (with `docker ps -a`), named like `docker-slam-run-[number]`. If you want to enter the container shell from the host machine, run `docker start [container-name]` to start the container and `docker exec -u ${USER} -it [container-name] /bin/bash`

### pip
Assume you are inside the container: 
```
python3 -m venv equiv_align_env
source equiv_align_env/bin/activate
cat requirements.txt | xargs -n 1 pip install
```
From now on, we assume that we are operating inside the container.

## Run training and testing
### Setup the data folder inside docker container
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



### Training
#### Modelnet
In docker container:
1. largescale training for modelnet: `sh scripts/train_modelnet_se3_largescale.sh`
2. training only on the airplane class for modelnet: `sh scripts/train_modelnet_se3_airplane_only.sh`

#### ETH3D and TUM format
In docker container:
1. largescale training for eth3d: `sh scripts/train_eth3d_se3_largescale.sh`

#### Tensorboard
In host:
1. Install tensorboard inside the previously created venv
```
source equiv_align_env/bin/activate
pip3 install tensorflow
```
launch the tensorboard:
`sh scripts/launch_tensorboard.sh`

### Testing
Note that when testing, batch_size = 1 for each gpu is recommended.
1. Test modelnet: `sh scripts/test_modelnet_se3.sh log/[pretrained-model-dir]/checkpoints/`
2. Test eth3d: `sh scripts/test_eth3d_se3_largescale.sh log/[pretrained-model-dir]/checkpoints/`





