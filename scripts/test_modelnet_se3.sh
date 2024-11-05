
PRETRAINED_CKPT_DIR=$1
python3 test_modelnet.py \
        exp_args --gpus "0" \
        --batch-size 1 \
        --num-workers 1 \
        --train-frac 0.6 \
        --val-frac 0.2 \
	--dataset-path /home/`whoami`/data/modelnet/EvenAlignedModelNet40PC/ \
        --naming-prefix modelnet_se3_test \
	--pretrained-model-dir $PRETRAINED_CKPT_DIR \
        --max-rotation-degree 90.0 \
        --max-translation-norm 0.0  \
	--crop-ratio 0.0 \
        net_args --num-optimization-iters 800 \
	--is-centerize  --is-learning-kernel --is-se3 \
	--is-gradient-ascent \
        --min-correlation-threshold 0.01 \
        --pose-regression-learning-rate  1e-6 \
	--use-full-inner-product \
        --use-normalized-pose-loss \
        --init-ell 1.0 \
        train_args --debug-mode --debug-mode \
        --num-training-optimization-iters 25 \
        --clip-norm 1e-1 \
        --learning-rate 1e-4 \
        --curriculum-list 90
