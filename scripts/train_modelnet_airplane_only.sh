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

python3 train_modelnet.py \
        exp_args --gpus "0" \
        --batch-size 8 \
        --num-workers 2 \
        --train-frac 0.6 \
        --val-frac 0.2 \
        --dataset-name 'ModelNet40Alignment' \
        --dataset-path "/home/`whoami`/data/modelnet/EvenAlignedModelNet40PC" --naming-prefix modelnet_airplane_only \
        --max-rotation-degree 30 \
        --max-translation-norm 0.25  \
        --modelnet-airplane-only \
        --is-shuffle \
        net_args --num-optimization-iters 200 \
	--encoder-type VnDgcnn \
        --is-se3 --is-learning-kernel  \
        --is-gradient-ascent \
        --min-correlation-threshold 0.01 \
        --pose-regression-learning-rate  1e-6 \
	--use-full-inner-product \
        --init-ell 0.25 \
	--use-normalized-pose-loss \
        train_args \
	--debug-mode \
        --num-training-optimization-iters 50 \
        --clip-norm 1e-1 \
	--num-epochs 10 \
        --learning-rate 1e-4 \
        --curriculum-list  5 15 30 60 90
