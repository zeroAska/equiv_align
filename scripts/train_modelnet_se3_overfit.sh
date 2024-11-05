
python3 train_modelnet.py \
        exp_args --gpus "0,1,2,3,4,5,6,7" \
        --batch-size 64 \
        --num-workers 4 \
        --train-frac 0 \
        --val-frac 1.0 \
	--is-overfitting \
        --dataset-name 'ModelNet40Alignment' \
        --dataset-path "/home/`whoami`/data/modelnet/EvenAlignedModelNet40PC" --naming-prefix modelnet_se3_overfit \
        --max-rotation-degree 30 \
        --is-shuffle \
        --max-translation-norm 1.0  \
        net_args --num-optimization-iters 200 \
	--encoder-type VnDgcnn \
        --is-se3 \
        --is-centerize  \
        --is-learning-kernel  \
        --is-gradient-ascent \
        --min-correlation-threshold 0.01 \
        --pose-regression-learning-rate  1e-6 \
	--use-full-inner-product \
        --use-normalized-pose-loss \
        --init-ell 1.0\
        train_args \
	--debug-mode \
        --num-training-optimization-iters 10 \
        --clip-norm 1e-1 \
        --learning-rate 1e-4 \
        --curriculum-list 30 
