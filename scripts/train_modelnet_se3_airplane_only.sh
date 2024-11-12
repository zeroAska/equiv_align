
python3 train_modelnet.py \
        exp_args --gpus "0" \
        --batch-size 4 \
        --num-workers 2 \
        --train-frac 0.6 \
        --val-frac 0.2 \
        --dataset-name 'ModelNet40Alignment' \
        --dataset-path "/home/`whoami`/data/modelnet/EvenAlignedModelNet40PC" --naming-prefix modelnet_se3_airplane_only \
        --max-rotation-degree 30 \
        --max-translation-norm 1.0  \
        --modelnet-airplane-only \
        --is-shuffle \
        net_args --num-optimization-iters 200 \
	--encoder-type VnDgcnn \
        --is-se3 \
        --is-learning-kernel  \
        --is-gradient-ascent \
        --min-correlation-threshold 0.01 \
        --pose-regression-learning-rate  1e-6 \
	--use-full-inner-product \
        --init-ell 1.0 \
	--use-normalized-pose-loss \
        train_args \
	--debug-mode \
        --num-training-optimization-iters 5 \
        --clip-norm 1e-1 \
	--num-epochs 10 \
        --learning-rate 1e-4 \
        --curriculum-list 1 5 10 20 30 40 50 60 70 80 90
