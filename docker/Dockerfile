FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=neoninteractive
ARG USER

# base tools
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y locales \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compute,utility,display



RUN apt update --fix-missing && apt install -y \
    build-essential vim git zip \
    imagemagick emacs sudo cargo 

# pip
RUN apt update && apt install -y python3-pip python3-tk python3-dev python3-venv 


# use current user
ARG USER_ID
ARG GROUP_ID
RUN if  ! grep -q $GROUP_ID /etc/group ; then  groupadd --gid $GROUP_ID $USER; fi 
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER
RUN adduser $USER sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

#RUN useradd -m docker && echo "docker:docker" | chpasswd
#USER docker


# clean
#WORKDIR /root
#ADD scripts/ /root/scripts/

USER $USER
# python packages
RUN pip3 install --upgrade pip && pip3 install pyyaml einops timm opencv-python matplotlib

# install pytorch
RUN pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117 


USER root
RUN apt update && apt install -y libsuitesparse-dev ffmpeg libsm6 libxext6 glibc-source libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev     libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk     libharfbuzz-dev libfribidi-dev libxcb1-dev  graphviz
