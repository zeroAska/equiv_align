## Installation
Assume your workspace directory is `equiv_vo/`

### NVIDIA Driver Setup on the cloud desktop
```
cd scripts/amazon
sh build_cloud_desktop.sh
```
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
Assume you are inside the container: `cat requirements.txt | xargs -n 1 pip install`
From now on, we assume that we are operating inside the container.

## Run training and testing
### Setup the data folder inside docker container
In docker container:
```
$ ls ~
equiv_vo data
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
### Pre-compiled data index pickle files for eth3d
WHen training on eth3d, do this following step first before starting the training, to save time from pre-processing data. In docker container:
```
cp data_loader/index_cache/eth3d/scene_cache.*.pickle data_loader/scene_cache/
cp data_loader/index_cache/eth3d/dataindex_cache.*.pickle data_loader/dataindex_cache/
```


### Training
#### Modelnet
In docker container:
1. Overfit training for modelnet: `sh scritps/train_modelnet_se3_overfit.sh`
2. largescale training for modelnet: `sh scripts/train_modelnet_se3_largescale.sh`
3. training only on the airplane class for modelnet: `sh scripts/train_modelnet_se3_airplane_only.sh`

#### ETH3D and TUM format
In docker container:
Note: eth3d training is not working correctly yet.
1. overfit training for eth3d: `sh scripts/train_eth3d_se3_overfit.sh`
2. largescale training for eth3d: `sh scripts/train_eth3d_se3_largescale.sh`

#### Tensorboard
In host:
1. Install tensorboard with venv
```
python3 -m venv equiv_vo_env
source equiv_vo_env/bin/activate
pip3 install tensorflow
```
launch the tensorboard:
`sh scripts/launch_tensorboard.sh`

### Testing
1. Test modelnet: `sh scripts/test_modelnet_se3.sh log/[pretrained-model-dir]/checkpoints/`
2. Test eth3d: `sh scripts/test_eth3d_se3_largescale.sh log/[pretrained-model-dir]/checkpoints/`




