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
python3 test.py \
        exp_args --gpus "0" \
        --batch-size 1 \
	--dataset-name eth3d \
	--dataset-path /home/`whoami`/equiv_align/data/eth3d/ \
        --num-workers 0 \
        --is-auto-split-dataset \
        --naming-prefix test_eth3d_se3_largescale \
	--pretrained-model-dir $1 \
	--max-rotation-degree 10.0 \
	--is-shuffle \
        --run-mode test \
        net_args --num-optimization-iters 400 \
	--encoder-type VnDgcnn --is-se3 \
	--is-centerize \
	--is-learning-kernel  \
        --is-gradient-ascent \
        --min-correlation-threshold 0.01 \
        --pose-regression-learning-rate  1e-7 \
	--use-full-inner-product \
	--use-normalized_pose_loss \
	--ell-learning-rate 1e-8 \
        --gram-mat-min-nonzero 1000 \
	--init-ell 0.2 \
	--min-ell 0.025 \
	--max-ell 0.25 \
        train_args \
	--num-epochs 100 \
        --num-training-optimization-iters 150 \
        --clip-norm 1e-2 \
        --learning-rate 1e-4 
