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

from data_loader.factory import create_datastream
from data_loader.tum import save_color_ply
import pypose as pp
import pypose.optim as ppos
from data_loader.factory import create_datastream
from model.option import gen_options, dump_args  # all the configs
from model.equiv_registration import EquivRegistration, LossModule
from model.metrics.metrics import pose_Fro_norm, pose_log_norm, translation_error, rotation_angle_error
from model.plot.tensorboard_plot import tensorboard_plot_grad, tensorboard_plot_pointset_pair
from test import test
from scipy.spatial.transform import Rotation as sciR
from model.utils import copy_new_leaf_tensor
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from torch import nn
import sys
import provider
import importlib
import shutil
import ipdb
import copy
from torch.utils.tensorboard import SummaryWriter
#from model import vnn
#from model.matcher.iterative import IterativeMatch
#from model.regression.inner_product_optimization import InnerProductOptimization
#from model.option import opt


#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = BASE_DIR
#sys.path.append(os.path.join(ROOT_DIR, 'models'))

def save_weights(checkpoints_dir, file_name,
                 epoch, mean_err,
                 equiv_net, optimizer):
    savepath = str(checkpoints_dir) +"/" + file_name
    print('Saving at %s'% savepath)
    state = {
        'epoch': epoch,
        'mean_err': mean_err if mean_err is not None else -1.0,
        'model_state_dict': equiv_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)
    


def train(opt):
    def log_string(string):
        logger.info(string)
        print(string)

    assert (opt.exp_args.run_mode == 'train')

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.exp_args.gpus

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    if (opt.exp_args.naming_prefix is not None):
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + str("-") + opt.exp_args.naming_prefix
    experiment_dir = Path(opt.exp_args.log_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, opt.net_args.encoder_type ))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(opt)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = opt.exp_args.dataset_path
    writer = SummaryWriter(log_dir=log_dir.absolute())
    
    trainDataLoader, valDataLoader, testDataLoader = create_datastream(opt)
    log_string('train data length: '+str( len(trainDataLoader)))
    log_string('val data length: '+str( len(valDataLoader)))
    log_string('test data length: '+str( len(testDataLoader)))
    if opt.exp_args.is_auto_split_dataset:
        trainDataLoader.dataset.save_index('train_')
        valDataLoader.dataset.save_index('val_')
        testDataLoader.dataset.save_index('test_')
    else:
        trainDataLoader.dataset.dataset.save_index('train_')
        valDataLoader.dataset.dataset.save_index('val_')
        testDataLoader.dataset.dataset.save_index('test_')
    
    '''MODEL LOADING'''
    equiv_net = EquivRegistration(opt).cuda()
    equiv_net.init_weights()
    equiv_net.dry_run(int(opt.exp_args.batch_size / torch.cuda.device_count()), opt.exp_args.num_point, writer)
    is_multi_gpu = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        equiv_net = nn.DataParallel(equiv_net).cuda()
        is_multi_gpu = True

    criterion = LossModule(opt).cuda()
    ckpt_path = os.path.join(opt.exp_args.pretrained_model_dir, 'best_model.pth')
    log_string(ckpt_path)

    try:
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        equiv_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if is_multi_gpu:
        net_single_gpu = equiv_net.module
    else: 
        net_single_gpu = equiv_net
    net_params = net_single_gpu.get_trainable_params()
    torch.autograd.set_detect_anomaly(True)

    if opt.train_args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            net_params,
            lr=opt.train_args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=opt.train_args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            net_params,
            lr=opt.train_args.learning_rate,
            momentum=0.9,
            weight_decay=opt.train_args.decay_rate
        )
    try:
        checkpoint = torch.load(ckpt_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=False)
        start_step = checkpoint['step'] if 'step' in checkpoint else 0
        log_string('Use previous optimizer state')
    except:
        log_string('No existing optimizer state, starting training from scratch...')
        start_step = 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    global_epoch = start_epoch
    global_step = start_step

    '''TRANING'''
    logger.info('Start training from epoch '+str(start_epoch))
    for epoch in range(start_epoch, opt.train_args.num_epochs):
        log_string('Epoch %d, total epoch %s:' % (epoch,  opt.train_args.num_epochs ))
        equiv_net.train()
        epoch_loss = 0
        total_batch_num = len(trainDataLoader)
        
        if opt.exp_args.is_auto_split_dataset:
            trainDataLoader.dataset.set_is_use_gt_init(epoch < opt.train_args.hold_pose_fix_epochs)
        else:
            trainDataLoader.dataset.dataset.set_is_use_gt_init(epoch < opt.train_args.hold_pose_fix_epochs)
        net_single_gpu.regression.set_is_holding_pose_fixed(epoch < opt.train_args.hold_pose_fix_epochs)
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
    
            equiv_net.zero_grad()
            equiv_net = equiv_net.train()
    
            pc1 = data['pc1'][:, :, :]
            pc2 = data['pc2'][:, :, :]
            Ts = data['T']
            if opt.net_args.is_se3:
                ts = data['t']
            Rs = Ts[:, :3, :3]
            pc1 = pc1.transpose(2, 1)
            pc2 = pc2.transpose(2, 1)

            #if True:
            if batch_id == 0:
                pc_save = torch.cat((pc1[0,:,:], pc2[0,:,:]), dim=-1).transpose(1,0).numpy()
                print("pc_save dim ",pc_save.shape)
                save_color_ply(pc_save, None, str(batch_id)+"_train_stacked.ply")
                save_color_ply(pc1[0,:,:].transpose(1,0).numpy(), None, str(batch_id)+"pc.ply")
                save_color_ply(pc2[0,:,:].transpose(1,0).numpy(), None, str(batch_id)+"pc.ply")

            if torch.isnan(data['pc1']).any() or \
               torch.isnan(data['pc2']).any()  or \
               torch.isnan(data['color1']).any() or \
               torch.isnan(data['color2']).any() or \
               torch.isnan(data['T']).any() or \
               torch.isnan(data['t']).any() or \
               torch.isnan(data['R']).any():
                log_string("Warning: training input data is nan")
                continue
                
            pc1, pc2, Rs = pc1.cuda(), pc2.cuda(), Rs.cuda()
            if opt.net_args.is_se3:
                T_init = pp.identity_SE3(pc1.shape[0]).cuda()
                Rs_tensor = torch.Tensor(sciR.from_matrix(Rs.cpu().numpy()).as_quat())
                ts_tensor = ts
                Ts_tensor = pp.SE3(torch.cat((ts_tensor, Rs_tensor), dim=-1)).cuda()
                ts = ts.cuda()                    
            else:
                T_init = pp.identity_SO3(pc1.shape[0]).cuda()
                Ts_tensor = pp.SO3(torch.Tensor(sciR.from_matrix(Rs.cpu().numpy()).as_quat())).cuda()                    
            T_init.requires_grad = True
            Ts_tensor.requires_grad = False
            T_init2 = copy_new_leaf_tensor(T_init, T_init.ltype)
                
            net_single_gpu.set_optimization_iters(1)

            instance_loss = 0
            for instance_iter in range(opt.train_args.num_training_optimization_iters):
                #print("Iter: ",instance_iter)
                pred, _, _, is_converged = equiv_net(pc1, pc2, T_init)
                if is_converged.detach().cpu().bool().any() == False:
                    equiv_net.zero_grad()
                    break
                loss = criterion(pred, Ts_tensor) /  pc1.shape[0] #/ pc1.shape[-1] / pc2.shape[-1]   #/ float(total_batch_num)
                if torch.isnan(loss).any():
                    log_string("Warning: training loss is nan")
                    optimizer.zero_grad()
                    equiv_net.zero_grad()
                    continue
                
                epoch_loss += loss
                ### TEST ######
                #net_single_gpu.encoder.zero_grad()
                ###############
                if not opt.train_args.is_unsupervised:
                    loss.backward(retain_graph=True)
                instance_loss += loss
                writer.add_scalar("LossStep/train", loss, #*  pc1.shape[-1] * pc2.shape[-1] , 
                                  (global_step)*opt.train_args.num_training_optimization_iters + instance_iter)
                if opt.train_args.clip_norm > 0:
                    nn.utils.clip_grad_norm_(net_params, opt.train_args.clip_norm)

                optimizer.step()
                optimizer.zero_grad()

                T_init = pred
            writer.add_scalar("LossInstance/train", instance_loss, #*  pc1.shape[-1] * pc2.shape[-1] , 
                              global_step)
            
            net_single_gpu.set_optimization_iters(opt.train_args.num_training_optimization_iters)
            print("Send T_init again to the trained network after one step")
            pred2, _, _, is_converged = equiv_net(pc1, pc2, T_init2)
            loss2 = criterion(pred2, Ts_tensor) / pc1.shape[0] 
            writer.add_scalar("LossUpdate/train", loss2-loss, global_step)            
            print("After current batch's training, loss changes from {} to {}".format(loss, loss2))
                
            if is_multi_gpu:
                tensorboard_plot_grad(writer, equiv_net.module.encoder, global_step, 'Encoder')
                tensorboard_plot_grad(writer, equiv_net.module.regression, global_step, 'Decoder')
                tensorboard_plot_grad(writer, equiv_net.module.matcher, global_step, 'Matcher')
                tensorboard_plot_grad(writer, criterion, global_step, 'Loss')
    
            else:
                tensorboard_plot_grad(writer, equiv_net.encoder, global_step, 'Encoder')
                tensorboard_plot_grad(writer, equiv_net.regression, global_step, 'Decoder')
                tensorboard_plot_grad(writer, equiv_net.matcher, global_step, 'Matcher')
                tensorboard_plot_grad(writer, criterion, global_step, 'Loss')
                    
            #if batch_id == 0:
            #    sampled_ind = int(np.random.random()*pc1.shape[0])
            #    tensorboard_plot_pointset_pair(writer,
            #                                   pc1[sampled_ind, :, :].permute(1,0),
            #                                   pc2[sampled_ind, :, :].permute(1,0),
            #                                   T_init[sampled_ind],
            #                                   pred[sampled_ind],
            #                                   'Train data batch 0 @ sample index '+str(sampled_ind)+' at epoch '+str(epoch),
            #                                   global_step,
            #                                   pp.SE3_type if opt.net_args.is_se3 else pp.SO3_type)
            #    #if epoch == 0:
                #    writer.add_graph()
                    
            global_step += 1

        writer.add_scalar("LossEpoch/train", epoch_loss, epoch)
    
        scheduler.step()

        if opt.train_args.is_saving_weight_every_epoch:
            save_weights(checkpoints_dir, 'train_epoch_{}.pth'.format(epoch),
                         epoch, None, equiv_net, optimizer)

        if epoch % opt.train_args.val_freq == 0:
    
            net_single_gpu.regression.set_is_holding_pose_fixed(False)
            if opt.exp_args.is_auto_split_dataset:
                valDataLoader.dataset.set_is_use_gt_init(False)
            else:
                valDataLoader.dataset.dataset.set_is_use_gt_init(False)
            err_val = test(equiv_net.eval(), valDataLoader, opt)
            err_dict = {
                'Frobenius Norm Error': torch.mean(torch.tensor(err_val['frobenius']['pred_err'])),
                'Matrix Log Norm Error': torch.mean(torch.tensor(err_val['log']['pred_err'])),
                'Rotation Angle Error': torch.mean(torch.tensor(err_val['angle']['pred_err'])),
                'Translation Norm Error': torch.mean(torch.tensor(err_val['translation_norm']['pred_err'])) if opt.net_args.is_se3 else float('nan')
            }
            #err_val  = test_tum(equiv_net.eval(), valDataLoader, opt)
            #err_dict = {}
            #for seq in err_val:
            #    err_dict['rpe_'+seq] = err_val[seq]['rpe']
            writer.add_scalars("Error/Val", err_dict, global_epoch)
            init_err_dict = {
                'Init Frobenius Norm Error': torch.mean(torch.tensor(err_val['frobenius']['init_err'])),
                'Init Matrix Log Norm Error': torch.mean(torch.tensor(err_val['log']['init_err'])),
                'Init Rotation Angle Error': torch.mean(torch.tensor(err_val['angle']['init_err'])),
                'Init Translation Norm Error': torch.mean(torch.tensor(err_val['translation_norm']['init_err'])) if opt.net_args.is_se3 else float('nan')
            }
            log_string('Val model: Init err: f-norm is %f, log-norm is %f, angle-err is %f, translation-err is %f'% (
                init_err_dict['Init Frobenius Norm Error'], init_err_dict['Init Matrix Log Norm Error'],
                init_err_dict['Init Rotation Angle Error'], init_err_dict['Init Translation Norm Error']
            ))
            writer.add_scalars("Error/Val", init_err_dict, global_epoch)
            
            #log_string('Val model: val Err on val dataset is: rpe translation is %f, rpe rotation is at epoch %d'% (errva
            #                                                                                                                                                  global_epoch))

            
                
            mean_err = torch.mean(torch.tensor(err_val['log']['pred_err']))
            if epoch == start_epoch:
                best_err = mean_err
            if mean_err <= best_err and not opt.exp_args.is_overfitting and not opt.train_args.is_skipping_test:
                best_err = mean_err
                best_epoch = global_epoch
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'mean_err': mean_err,
                    'model_state_dict': equiv_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                net_single_gpu.set_optimization_iters(opt.net_args.num_optimization_iters)
                if opt.exp_args.is_auto_split_dataset:
                    testDataLoader.dataset.set_is_use_gt_init(False)
                else:
                    testDataLoader.dataset.dataset.set_is_use_gt_init(False)
                err_test = test(equiv_net.eval(), testDataLoader, opt)
                err_dict = {
                    'Frobenius Norm Error': torch.mean(torch.tensor(err_test['frobenius']['pred_err'])),
                    'Matrix Log Norm Error': torch.mean(torch.tensor(err_test['log']['pred_err'])),
                    'Rotation Angle Error': torch.mean(torch.tensor(err_test['angle']['pred_err'])),
                    'Translation Norm Error': torch.mean(torch.tensor(err_test['translation_norm']['pred_err'])) if opt.net_args.is_se3 else float('nan')}
                writer.add_scalars("Error/Test", err_dict, best_epoch)
                init_err_dict = {
                    'Init Frobenius Norm Error': torch.mean(torch.tensor(err_test['frobenius']['init_err'])),
                    'Init Matrix Log Norm Error': torch.mean(torch.tensor(err_test['log']['init_err'])),
                    'Init Rotation Angle Error': torch.mean(torch.tensor(err_test['angle']['init_err'])),
                    'Init Translation Norm Error': torch.mean(torch.tensor(err_test['translation_norm']['init_err'])) if opt.net_args.is_se3 else float('nan')
                }
                writer.add_scalars("Error/Test", init_err_dict, best_epoch)
                log_string('Test model: Init err: f-norm is %f, log-norm is %f, angle-err is %f, translation-err is %f'% (
                    init_err_dict['Init Frobenius Norm Error'], init_err_dict['Init Matrix Log Norm Error'],
                    init_err_dict['Init Rotation Angle Error'], init_err_dict['Init Translation Norm Error']
                ))
                
                
                log_string('Best val model: Test Err on test dataset is: Best Err: f-norm is %f, log-norm is %f, angle-err is %f, translation-err is %f, at epoch %d'% (err_dict['Frobenius Norm Error'], err_dict['Matrix Log Norm Error'],
                                                                                                                                                                        err_dict['Rotation Angle Error'], err_dict['Translation Norm Error'],
                                                                                                                                                                        best_epoch))

            writer.flush()
            savepath = str(checkpoints_dir) +"/" + str(epoch)+ '_model.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': global_epoch,
                'step': global_step,
                'mean_err': mean_err,
                'model_state_dict': equiv_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            
        global_epoch += 1
    
    logger.info('End of training...')
    

if __name__ == '__main__':
    #args = parse_args()
    opt = gen_options()
    train(opt)
