python3 test.py \
        exp_args --gpus "0" \
        --batch-size 2 \
	--dataset-name eth3d \
	--dataset-path /home/`whoami`/data/eth3d/ \
        --num-workers 2 \
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
