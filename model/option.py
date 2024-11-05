'''
Source: vgtk https://github.com/minghanz/E2PN/blob/398666c4e537d3d4b009bb824e4d9865f8edb2b0/vgtk/vgtk/app/parse_config.py
Refactored by Ray Zhang
'''

import argparse
from argparse import Namespace
import os
import ipdb
import getpass


class HierarchyArgmentParser():
    
    def __init__(self, flatten_args=[]):
        super(HierarchyArgmentParser, self).__init__()
        self.flatten_args = flatten_args
        self.parser = argparse.ArgumentParser()
        self.sub = self.parser.add_subparsers()
        self.parser_list = {}

    def add_parser(self, name):
        args = self.sub.add_parser(name)
        self.parser_list[name] = args
        return args

    def parse_args(self):
        #opt_all, _ = self.parser.parse_known_args()
        opt_all = Namespace(**{})
        for name, parser in self.parser_list.items():
            opt, _ = parser.parse_known_args()
            if name in self.flatten_args:
                setattr(opt_all, name, opt)
                for key, value in vars(opt).items():
                    print(key, value)
                #    setattr(opt_all, key, value)
            else:
                raise RuntimeError("parse_args: unknown key "+str(name))
            #setattr(opt_all, name, opt)
        return opt_all


def dump_args(opt):
    args = {}
    for k, v in vars(opt).items():
        if isinstance(v, argparse.Namespace):
            args[k] = vars(v)
        else:
            args[k] = v
    return args

def gen_options():

    parser = HierarchyArgmentParser(flatten_args=['exp_args','net_args','train_args'])

    ### exp logging args ###
    exp_args = parser.add_parser("exp_args")
    exp_args.add_argument("-n", '--dataset-name', type=str, default='ModelNet40Alignment', help="the name of the dataset")
    exp_args.add_argument('-d', '--dataset-path', type=str, default="/home/"+getpass.getuser()+"/data/modelnet/EvenAlignedModelNet40PC", help="the full path for the dataset folder")
    exp_args.add_argument('--seq',type=str, default=None, help="if not None, using the specific sequence for testing")
    exp_args.add_argument('--gpus', type=str, default='0',
                          help='environment variable for CUDA_VISIBLE_DEVICES')
    exp_args.add_argument('--log-dir', type=str, default='log/', help='result logging folder')
    exp_args.add_argument('--num-point', type=str, default=1024, help='number of input points after downsampling from the dataloader')
    exp_args.add_argument('--batch-size', type=int, default=8, help='batch size for training or testing')
    exp_args.add_argument('--num-workers', type=int, default=8, help='number of workers for training or testing')
    exp_args.add_argument('--pretrained-model-dir', type=str, default='checkpoints/', help='path to models (the root of all outputs)')
    #exp_args.add_argument('--pretrained-model-dir', type=str, default='log/2023-07-18_19-15-train_multi_backprop_large_scale/checkpoints/', help='path to models (the root of all outputs)')
    exp_args.add_argument('-s', '--seed', type=int, default=2913,
                          help='random seed')
    exp_args.add_argument('--run-mode', type=str, default='train',
                          help='train | val  | test | eval_traj')
    exp_args.add_argument('--is-normal-channel', action='store_true',
                          help='is using surface normal in the channel, only for point cloud dataset')
    exp_args.add_argument('--rot-augmentation', type=str, default='so3',
                          help='options  of rotation augmentation include e[so3 | z | None]')
    exp_args.add_argument('--noise-augmentation', type=float, default=float(),
                          help='options  of noise augmentation')
    exp_args.add_argument('--outlier-augmentation', type=float, default=float(),
                          help='options  of outlier augmentation percent')
    exp_args.add_argument('--train-frac', type=float, default=0.6,
                          help='use less than 1.0 to test if the network can be trained with fewer data. ')
    exp_args.add_argument('--val-frac', type=float, default=0.2,
                          help='use less than 1.0 to test if the network can be trained with fewer data. ')
    exp_args.add_argument('--is-overfitting', action='store_true', help='whether use a tiny dataset for overfitting test. the training and the test dataset will be the same')
    exp_args.add_argument('--max-rotation-degree', type=float, default=0.0, help='max ground truth rotation angle')
    exp_args.add_argument('--max-translation-norm', type=float, default=0.0, help='max ground truth translation')
    exp_args.add_argument('--naming-prefix',type=str, default=None, help='addional naming prefix to the log folder')
    exp_args.add_argument('--odom-covis-thresh',type=float, default=0.95, help='the overlap threshold between two frames in a traj to be considered a match')
    exp_args.add_argument('--modelnet-airplane-only', action='store_true', help='use only the unsymmetric plane category in the modelnet dataset')
    exp_args.add_argument('--modelnet-half-categories', action='store_true', help='use only the unsymmetric plane category in the modelnet dataset')
    exp_args.add_argument('--modelnet-symmetric-categories', action='store_true', help='use only the symmetric categories in the modelnet dataset')
    exp_args.add_argument('--modelnet-selected-categories', nargs='+', type=str, default=[], help='list of the selected modelnet categories')    
    exp_args.add_argument('--is-auto-split-dataset', action='store_true', help='whether manually split train/val/tst dataset, or use a predefined train/test split')
    exp_args.add_argument('--baseline-type', type=str, default=None, help='choice of baseline when running test_baseline.py')
    exp_args.add_argument('--is-shuffle',action='store_true', help='is shuffling point clouds')
    exp_args.add_argument('--skip-random-permutation', action='store_true', help='is skipping the random permuation of input points')
    exp_args.add_argument('--crop-ratio', type=float, default=0.0, help='is randomly cropping a portion of the input pair of point clouds')
    exp_args.add_argument('--use-gt-init', action='store_true', help='is using ground truth pose as init value')
    exp_args.add_argument('--is-inlier_only', action='store_true', help='is using inlier only')
    exp_args.add_argument('--edge-only',action='store_true',help='is downsample with edge only?')

    #exp_args.add_argument('--is-user-specified-trainvaltest', action='store_true', help='whehter let use decide what are the train/val/test dataset')
    exp_args.add_argument('--num-eval-times', type=int, default=1, help='number of times to sweep through the eval data')
    #exp_args.add_argument('--is-eval-traj',action='store_true',help='is eval full traj')    
    exp_args.add_argument('--eval-traj-seq', nargs='+', type=str, default=[], help='besides the pairwise errors, whether the trajectory-wise evaluations are performed evaluated as well. The traj name to eval are listed here.')
    exp_args.add_argument('--is-eval-traj-wise-metric', action='store_true', help='whether the full traj metric are stored as well')

    ### network config args ###
    net_args = parser.add_parser("net_args")
    net_args.add_argument('-e', '--encoder-type', type=str, 
                          default="VnDgcnn",
                          help='choice of encoder: VnDgcnn | IdentityEncoder | VnDgcnnSE3')
    net_args.add_argument('--is-se3', action='store_true', help='is enabling SE3 equivariance')
    net_args.add_argument('--is-centerize', action='store_true', help='is centerizing the input point clouds')
    net_args.add_argument('--is-normalize',action='store_true', help='whether normalizing the point cloud inputs to [-1,1]')
    net_args.add_argument('-m', '--matcher-type', type=str, default='IterativeMatch',
                          help='choice of matcher')
    #net_args.add_argument('--transformation-type', type=str, default='SO3', help='choices: SO3 | SE3')
    net_args.add_argument('-r', '--regression-type', type=str, default='InnerProductOptimization', help='choice of regression')
    net_args.add_argument('-k', '--n-knn', type=int, default=20, help='num of neighbors in the encoder')
    #net_args.add_argument('--is-coord-conv', action='store_true', help='is doing graph conv on the coord instead of the feature space' )
    net_args.add_argument('-l', '--init-ell', type=float, default=0.5, help='initial kernel lengthscale')
    net_args.add_argument('--min-ell', type=float, default=0.05, help='initial kernel lengthscale')
    net_args.add_argument('--max-ell', type=float, default=2.0, help='initial kernel lengthscale')
    net_args.add_argument('--kernel-type', type=str, default='CosKernel', help='choice of kernel')
    net_args.add_argument('--is-learning-kernel', action='store_true',
                          help='whether learn the kernel or use constant kernel')
    net_args.add_argument('--is-using-single-kernel', action='store_true',
                          help='whether using a single kernel for both the coordinates and the steerable feature maps')
    net_args.add_argument('-i', '--num-optimization-iters', type=int, default=10,
                          help='number of iterations during optimization')
    net_args.add_argument('--min-correlation-threshold', type=float, default=0.01,
                          help='the smallest value for deeciding if the correlation is zero')
    net_args.add_argument('--is-visualizing', action='store_true',
                          help='whether using matplotlib to visualize the feature maps and the matches, and the rotated poses and save them into png/gif files')
    net_args.add_argument('--is-second-order-optimization', action='store_true')
    net_args.add_argument('--regression-residual-thresh', type=float, default=1e-6,
                          help='the residual threshold to stop')
    net_args.add_argument('--trust-region-radius', type=float, default=1e4, help='trust region radius')
    net_args.add_argument('--kanchor', type=int, default=12,
                          help='number of finite rotation anchors if using rotation discretization')
    net_args.add_argument('--is-gradient-ascent', action='store_true')
    net_args.add_argument('--pose-regression-learning-rate',type=float,default=1e-5)
    net_args.add_argument('--ell-learning-rate',type=float,default=1e-7)
    net_args.add_argument('--gram-mat-min-nonzero',type=int,default=10000)
    net_args.add_argument('--predict-non-iterative', action='store_true', help='use non-iterative prediction in the regressor')
    net_args.add_argument('--use-feature-pyramid',action='store_true', help='is using feature pyramid, otherwise just use the last layer')
    net_args.add_argument('--use-full-inner-product',action='store_true', help='is calculating the self-correlation as well')
    net_args.add_argument('--use-identity-feature-map',action='store_true', help='only uses coordinates without features. reduced to the original cvo')
    net_args.add_argument('--use-normalized-pose-loss', action='store_true', help='is normalizing the pose loss with batch size on each device')
    net_args.add_argument('--is-updating-coord', action='store_true', help='is updating coordinates in encoder')
    net_args.add_argument('--is-holding-pose-fixed', action='store_true', help='whether holding the pose unchanged')

    ### training args ###
    train_args = parser.add_parser("train_args")
    train_args.add_argument('--val-freq', type=int, default=5, help='validation frequency during training')
    train_args.add_argument('--debug-mode', action='store_true',
                            help='if specified, train with a certain debug procedure')
    train_args.add_argument('--rot-ref-tgt', action="store_true",
                            help='regress rotation with tgt as reference if set true')
    train_args.add_argument('--num-epochs', type=int, default=30, help='Number of training epochs per init angle' )
    train_args.add_argument('--check-loss-update-each-step',action='store_true', help='is checking if the loss decreases after backprop, with the same input')
    train_args.add_argument('--optimizer', type=str, default='Adam', help='optimizer type')
    train_args.add_argument('--learning-rate',type=float, default=1e-3)
    train_args.add_argument('--decay-rate',type=float, default='0.9')
    train_args.add_argument('--clip-norm', type=float, default=0.01)
    train_args.add_argument('--loss-type', type=str, default='matrix_logarithm', help='choices: matrix_logarithm | frobenius_norm')
    train_args.add_argument('--num-training-optimization-iters',type=int, default=1, help='number of training backprop iters per data instance ')
    train_args.add_argument('--curriculum-list', nargs='+', type=float, default=[], help='list of the maximum training init angles for training')
    train_args.add_argument('--is-unsupervised', action='store_true', help='is using unsuperised training')
    train_args.add_argument('--hold-pose-fix-epochs', type=int, default=0, help='number of epochs that hold gt pose fixed')
    train_args.add_argument('--is-saving-weight-every-epoch', action='store_true', help='is stroig weight for every iteration')
    train_args.add_argument('--is-skipping-test', action='store_true', help='is skipping testing stage')

    opt = parser.parse_args()        
    return opt
