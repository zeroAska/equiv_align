"""
Author: Ray Zhang
"""
#from data_loader.ModelNetDataLoader import ModelNetDataLoader
import pypose as pp
import time
import torch.nn as nn
import pypose.optim as ppos
from data_loader.factory import create_datastream
from model.option import gen_options, dump_args  # all the configs
from model.equiv_registration import EquivRegistration
from model.metrics.metrics import pose_Fro_norm, pose_log_norm, translation_error, rotation_angle_error
from model.utils import filter_ckpt_state_dict
import numpy as np
#from scipy.spatial.transform import Rotation as scipy_rotation
from scipy.spatial.transform import Rotation as sciR

import os
import torch
import logging
from tqdm import tqdm
import sys
import pdb, ipdb
import importlib
#from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = BASE_DIR
#sys.path.append(os.path.join(ROOT_DIR, 'models'))


def test(model, loader, opt):

    torch.set_printoptions(precision=4, sci_mode=False)
    
    result = {
        'log': {'op': pose_log_norm, 'init_err': [], 'pred_err': []},
        'frobenius': {'op': pose_Fro_norm, 'init_err': [], 'pred_err': []},
        'angle': {'op': rotation_angle_error, 'init_err': [], 'pred_err': []},
        'pose_pair_list': []
    }
    if opt.net_args.is_se3:
        result['translation_norm'] = { 'op': translation_error, 'init_err': [], 'pred_err': []}
        result['ltype'] = pp.SE3_type
    else:
        result['ltype'] = pp.SO3_type
    #if opt.exp_args.is_eval_traj:
        
    total_time = 0
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        model.zero_grad()
        pc1 = data['pc1'][:, :, :]
        pc2 = data['pc2'][:, :, :]
        print("pc1.shape: ",pc1.shape)
        #names = data['fn']
        Ts = data['T']
        #Rs = data['R']
        if opt.net_args.is_se3:
            t_label = data['t']
        #R_labels = data['R_label']
        #anchor_labels = data['anchor_label']
        batch_size = pc1.shape[0]
        #if opt.exp_args.is_eval_traj:
        #    seq_name = data['seq']
        #    pc1_id = data['pc1_id']
        #    pc2_id = data['pc2_id']
            

        #pdb.set_trace()
        
        #if opt.exp_args.rot_augmentation == 'z':
        #    trot = RotateAxisAngle(angle=torch.rand(points.shape[0])*360, axis="Z", degrees=True)
        #elif opt.exp_args.rot_augmentation == 'so3':
        #    trot = Rotate(R=random_rotations(points.shape[0]))
        #points = trot.transform_points(points)
        #pdb.set_trace()
        pc1 = pc1.transpose(2, 1)
        pc2 = pc2.transpose(2, 1)
        pc1, pc2, Ts = pc1.cuda(), pc2.cuda(), Ts.cuda()
        if opt.net_args.is_se3:
            T_init = pp.identity_SE3(pc1.shape[0]).cuda()
            Rs_tensor = torch.Tensor(sciR.from_matrix(Ts[:,:3,:3].cpu().numpy()).as_quat())
            ts_tensor = t_label
            Ts_tensor = pp.SE3(torch.cat((ts_tensor, Rs_tensor), dim=-1)).cuda()
            #ts = ts.cuda()                    
        else:
            T_init = pp.identity_SO3(pc1.shape[0]).cuda()
            Ts_tensor = pp.SO3(torch.Tensor(sciR.from_matrix(Ts[:,:3,:3].cpu().numpy()).as_quat()))
        print("gt is \n{}".format(Ts_tensor))
        T_init.requires_grad = True
        Ts_tensor.requires_grad = False
     
        T_init_copy = T_init.cpu().clone()

        tic = time.perf_counter()
        pred, _, _, is_converged = model(pc1, pc2, T_init)
        toc = time.perf_counter()
        total_time += toc - tic

        for err_type in result:
            #ipdb.set_trace()
            if not isinstance(result[err_type], dict) or ('op' not in result[err_type]):
                continue
            #try:
            init_err = result[err_type]['op'](T_init_copy, Ts_tensor.cpu(), result['ltype'])
            #except TypeError as e:
            #    print("Exception {e}")
            #    ipdb.set_trace()
            pred_err = result[err_type]['op'](pred.cpu(), Ts_tensor.cpu(), result['ltype'])
            result[err_type]['init_err'].extend(init_err)
            result[err_type]['pred_err'].extend(pred_err)
        
        result['pose_pair_list'].append({'T_init': T_init_copy,
                                         'T_pred': pred,
                                         'T_gt': Ts_tensor})
            
        print("pose changes from {} to\n{}".format(T_init_copy, pred.cpu()))
        print("ground truth pose is {}".format(Ts_tensor.cpu()))
        print("mean Matrix Logrithm error changes from {} to {}".format(torch.Tensor(result['log']['init_err'][-batch_size:]).mean().item(),
                                                                        torch.Tensor(result['log']['pred_err'][-batch_size:]).mean().item() ))
        print("mean frobenius norm error changes from {} to {}".format( torch.Tensor(result['frobenius']['init_err'][-batch_size:]).mean().item(),
                                                                        torch.Tensor(result['frobenius']['pred_err'][-batch_size:]).mean().item()) )
        
        print("==============================================================")

    print("avg running time per frame is ", total_time / len(loader))

    return result


def main(opt):
    def log_string(str):
        logger.info(str)
        print(str)

    torch.autograd.set_detect_anomaly(True)

    print(dump_args(opt))

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.exp_args.gpus

    '''CREATE DIR'''
    experiment_dir = opt.exp_args.log_dir #'log/cls/' + args.log_dir

    '''LOG'''
    #args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(dump_args(opt))
    
    

    '''DATA LOADING'''
    log_string('Load dataset ...')
    #DATA_PATH = opt.exp_args.dataset_path #'data/modelnet40_normal_resampled/'
    #TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=opt.exp_args.num_point, split='test', normal_channel=args.normal)
    #testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=opt.exp_args.batch_size, shuffle=False, num_workers=4)
    _, _, testDataLoader = create_datastream(opt)

    '''MODEL LOADING'''
    equiv_model = EquivRegistration(opt)
    is_multi_gpu = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        equiv_model = nn.DataParallel(equiv_model).cuda()
        is_multi_gpu = True
    
    checkpoint = torch.load(os.path.join(opt.exp_args.pretrained_model_dir, 'best_model.pth'))
    if is_multi_gpu:
        model_dict = filter_ckpt_state_dict(checkpoint['model_state_dict'], equiv_model.module.state_dict(), ['matcher.T', 'regression.matcher.T'])
        equiv_model.load_state_dict(model_dict, strict=False)
    else:
        model_dict = filter_ckpt_state_dict(checkpoint['model_state_dict'], equiv_model.state_dict(), ['matcher.T', 'regression.matcher.T'])
        equiv_model.load_state_dict(model_dict, strict=False)

    '''START TEST'''
    #with torch.no_grad():
    equiv_model = equiv_model.cuda().eval()    
    err_test = test(equiv_model, testDataLoader, opt)
    #for err in registration_error:
    log_string('Test finish: total error log size is {}'.format(len(err_test['log']['pred_err'])))
    #import ipdb; ipdb.set_trace()
    err_dict = {
        'Frobenius Norm Error': torch.mean(torch.tensor(err_test['frobenius']['pred_err'])),
        'Matrix Log Norm Error': torch.mean(torch.tensor(err_test['log']['pred_err'])),
        'Rotation Angle Error': torch.mean(torch.tensor(err_test['angle']['pred_err'])),
        'Translation Norm Error': torch.mean(torch.tensor(err_test['translation_norm']['pred_err'])) if opt.net_args.is_se3 else float('nan')
    }
    init_err_dict = {
        'Frobenius Norm Error': torch.mean(torch.tensor(err_test['frobenius']['init_err'])),
        'Matrix Log Norm Error': torch.mean(torch.tensor(err_test['log']['init_err'])),
        'Rotation Angle Error': torch.mean(torch.tensor(err_test['angle']['init_err'])),
        'Translation Norm Error': torch.mean(torch.tensor(err_test['translation_norm']['init_err'])) if opt.net_args.is_se3 else float('nan')
    }
    max_err_dict = {
        'Frobenius Norm Error': torch.max(torch.tensor(err_test['frobenius']['pred_err'])),
        'Matrix Log Norm Error': torch.max(torch.tensor(err_test['log']['pred_err'])),
        'Rotation Angle Error': torch.max(torch.tensor(err_test['angle']['pred_err'])),
        'Translation Norm Error': torch.max(torch.tensor(err_test['translation_norm']['pred_err'])) if opt.net_args.is_se3 else float('nan')
    }
    std_err_dict = {
        'Frobenius Norm Error': torch.std(torch.tensor(err_test['frobenius']['pred_err'])),
        'Matrix Log Norm Error': torch.std(torch.tensor(err_test['log']['pred_err'])),
        'Rotation Angle Error': torch.std(torch.tensor(err_test['angle']['pred_err'])),
        'Translation Norm Error': torch.std(torch.tensor(err_test['translation_norm']['pred_err'])) if opt.net_args.is_se3 else float('nan')
    }



    log_string('Init result of %d data: Test Err : f-norm is %f, log-norm is %f, angle-err is %f, translation-err is %f'% (len(err_test['frobenius']['pred_err']),
                                                                                                                           init_err_dict['Frobenius Norm Error'], init_err_dict['Matrix Log Norm Error'],
                                                                                                                           init_err_dict['Rotation Angle Error'], init_err_dict['Translation Norm Error']
                                                                                                                           ))

    log_string('Test result of %d data: Test Err : f-norm is %f, log-norm is %f, angle-err is %f, translation-err is %f'% (len(err_test['frobenius']['pred_err']),
                                                                                                                           err_dict['Frobenius Norm Error'], err_dict['Matrix Log Norm Error'],
                                                                                                                           err_dict['Rotation Angle Error'], err_dict['Translation Norm Error']
                                                                                                                           ))
    log_string('Test result of %d data: Max Test Err : f-norm is %f, log-norm is %f, angle-err is %f, translation-err is %f'% (len(err_test['frobenius']['pred_err']),
                                                                                                                           max_err_dict['Frobenius Norm Error'], max_err_dict['Matrix Log Norm Error'],
                                                                                                                           max_err_dict['Rotation Angle Error'], max_err_dict['Translation Norm Error']
                                                                                                                           ))
    log_string('Test result of %d data: STD Test Err : f-norm is %f, log-norm is %f, angle-err is %f, translation-err is %f'% (len(err_test['frobenius']['pred_err']),
                                                                                                                           std_err_dict['Frobenius Norm Error'], std_err_dict['Matrix Log Norm Error'],
                                                                                                                           std_err_dict['Rotation Angle Error'], std_err_dict['Translation Norm Error']
                                                                                                                           ))


if __name__ == '__main__':
    #args = parse_args()
    opt = gen_options()
    main(opt)
