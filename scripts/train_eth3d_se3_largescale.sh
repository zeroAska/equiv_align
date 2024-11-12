# This file is part of EquivAlign.
# 
# Copyright [2024] [Authors of Paper: Correspondence-free SE(3) point cloud registration in RKHS via unsupervised equivariant learning]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Author Email: <Ray Zhang rzh@umich.edu>

python3 train_tum.py \
        exp_args --gpus "0" \
        --batch-size  1 \
	--num-workers 0 \
	--dataset-name eth3d \
	--dataset-path /home/`whoami`/data/eth3d/ \
        --is-auto-split-dataset \
        --naming-prefix eth3d_se3_largescale \
        --is-shuffle \
	--max-rotation-degree 10 \
        net_args --num-optimization-iters 200 \
	--encoder-type VnDgcnn \
        --is-se3 \
        --is-centerize  \
        --is-learning-kernel  \
        --is-gradient-ascent \
        --min-correlation-threshold 0.01 \
        --pose-regression-learning-rate  1e-6 \
	--use-full-inner-product \
	--use_normalized_pose_loss \
        --init-ell 0.2 \
	--min-ell 0.025 \
	--max-ell  0.25 \
	--ell-learning-rate 1e-8 \
        --gram-mat-min-nonzero 1000 \
        train_args \
        --num-training-optimization-iters 150 \
        --clip-norm 1e-2 \
	--num-epochs 100 \
        --learning-rate 1e-4
