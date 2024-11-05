
python3 train_modelnet.py \
        exp_args --gpus "0,1,2,3,4,5,6,7" \
        --batch-size 64 \
        --num-workers 4 \
        --train-frac 0.6 \
        --val-frac 0.2 \
        --dataset-name 'ModelNet40Alignment' \
        --dataset-path "/home/`whoami`/data/modelnet/EvenAlignedModelNet40PC" --naming-prefix modelnet_se3_largescale \
        --max-rotation-degree 30 \
        --max-translation-norm 1.0  \
        --is-shuffle \
        net_args --num-optimization-iters 25 \
	--encoder-type VnDgcnn \
        --is-se3 \
        --is-centerize  \
        --is-learning-kernel  \
        --is-gradient-ascent \
        --min-correlation-threshold 0.01 \
        --pose-regression-learning-rate  1e-6 \
	--use-full-inner-product \
        --init-ell 1.0 \
	--use-normalized-pose-loss \
        train_args \
	--debug-mode \
        --num-training-optimization-iters 10 \
        --clip-norm 1e-1 \
	--num-epochs 10 \
        --learning-rate 1e-4 \
        --curriculum-list 1 5 15 30 60 90
