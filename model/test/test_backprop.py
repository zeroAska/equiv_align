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
import torch
from torch import nn
from torch.nn.functional import normalize
import pypose as pp
import numpy as np
import pdb
import theseus as th

def test_backprop():
    a0 = torch.Tensor([1,2,3,4]) 
    a = a0 + 1
    a = normalize(a, dim=0)
    b = normalize(torch.Tensor([4,3,2,1]), dim=0)

    #assert(torch.equal(a.unsqueeze(0).transpose(0,1), a), 1.0)
    #assert(torch.equal(b.unsqueeze(0).transpose(0,1), b), 1.0)
    #assert(torch.equal(torch.eye(3), pp.SO3(a).Inv().matrix() @ (pp.SO3(a).matrix())))

    a0.requires_grad = True
    a.requires_grad = True
    b.requires_grad = True
    c = (pp.SO3(a).Inv() @ pp.SO3(b)).Log().norm()
    c.backward(retain_graph=True)
    print("c.grad: ",c.grad)
    print("a0.grad: ",a0.grad)
    print("a.grad: ",a.grad)
    print("b.grad: ",b.grad)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
    def forward(self, a, idd):
        assert a.is_leaf
        return (a.Inv() @ idd).Log().norm()
        

def test_iterative():
    a = pp.SO3(normalize(torch.Tensor([1,2,3,4]), dim=0))
    a.requires_grad = True
    idd = pp.identity_SO3()

    print(idd @ idd.Inv())

    net = SimpleNet()
    opt = torch.optim.SGD([a], lr=1e-2, momentum=0.9)
    for i in range(50):
        print("============ iter {} ==============".format(i))
        opt.zero_grad()
        loss = net(a, idd)
        loss.backward()

        a_old = a.clone()
        opt.step()
        print("a.grad: ", a.grad)        
        print("a: ",a)
        print("a-a_old=",a-a_old)

        
        
import torch
import torch.nn
import pypose as pp
class DataParallelModel(nn.Module):
    def __init__(self):
        super(DataParallelModel, self).__init__()
        #self.register_parameter('T_so3', None)

    def forward(self, T, target):
        

        #if (self.T_so3 is None):
        #    self.T_so3 = pp.Parameter(pp.SO3(T))
        if not isinstance(T, pp.LieTensor):
            T = pp.so3(T)
            #assert(T.is_leaf)        
        #y_so3 = pp.SO3(target)

        #T_so3 = pp.SO3(T_so3)
        #x = pp.SO3(x)
        #T_so3 = T

        #assert(isinstance(T_so3, pp.LieTensor))
        return pp.Exp(T) @ target

    

def test_parallel():
    batch_size = torch.cuda.device_count() * 200
    a = pp.identity_so3().cuda()
    a.requires_grad = True

    a_init = a.clone()

    data = torch.rand(batch_size, 3).cuda() * 10

    gt_SO3 = pp.SO3([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)])
    #
    gt_so3 = pp.Log( gt_SO3).cuda().unsqueeze(0).expand(batch_size, 3)
    gt_SO3_batch = gt_SO3.cuda().unsqueeze(0).expand(batch_size, 4)
    data_new =  gt_SO3_batch @ data
    print("gt_so3 shape ",gt_so3.shape)
        
    net = DataParallelModel()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net).cuda()
        optimizer = torch.optim.SGD([a], lr=1e-4)
        for i in range(200):
            #result = pp.SO3(net(a_batch, data))
            a_batch = a.unsqueeze(0).expand(batch_size, a.shape[0])

            result =net(a_batch, data)
            #loss = (result.Inv() @ data_new).Log().norm()
            loss = (result -data_new).norm()
            loss.backward()

            #a = a - a.grad * 1e-4
            optimizer.step()
            #a.zero_grad()
            net.zero_grad()
            optimizer.zero_grad()

            print("a @ iter {} is {}, loss is {}".format(i,a, loss))
        #print("a * data - data_rotated is {}".format(a @ data -data_rotated))
        print("a_init is {}".format(a_init))
        print("a is {}".format(a))
        print("gt is {}".format(gt_so3[0,:]))
        #print("gt * a.Inv() is {}".format(gt_SO3[0] @ a.Exp()))


def test_parallel_forward():
    a =(torch.Tensor([1,2,3])) #pp.randn_SO3().cuda()
    a.requires_grad = True 
    
    net = DataParallelModel()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net).cuda()
        optimizer = torch.optim.SGD([a], lr=1e-1)
        data = torch.rand(4,3).cuda()
        a = a.unsqueeze(0).expand(4, a.shape[0])
        result = net(data, a)


class DataParallelTheseusModel(nn.Module):
    def __init__(self):
        super(DataParallelTheseusModel, self).__init__()
        #self.register_parameter('T', None)

    def forward(self,
                x,
                T_so3):


        #self.T = nn.Parameter(T)
        #print("type of T {} is {}".format(T_so3, type(T_so3)))        
        #T_so3 = pp.SO3(T_so3)
        #x = pp.SO3(x)
        #T_so3 = T

        #assert(isinstance(T_so3, pp.LieTensor))
        return T_so3.update(x)#T_so3 @ x

def loss_fn(optim_vars, aux_vars):
    g = optim_vars[0]
    data, data_new = aux_vars
    result = (g.rotate(data) - data_new).tensor.norm(dim=-1).unsqueeze(1).transpose(0,1)
    print(result.shape)
    return result
    
def test_data_parallel_theseus():

    batch_size = torch.cuda.device_count()
    num_data = 100
    g = th.SO3(quaternion=torch.Tensor([[0,0,0,1.0]]), name='g')
    print(g.shape)
    g_init = th.SO3(quaternion=torch.Tensor([0,0,0,1.0]), name='g_init')

    data_tensor = torch.rand(batch_size, num_data, 3) * 10
    data_tensor_vec = data_tensor#.view(-1, 3)
    #data = data_tensor_vec
    data = th.Vector(tensor=data_tensor_vec, name='x')
    gt = th.SO3(quaternion=torch.Tensor([0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]), name='gt')

    data_new = gt.rotate(data)
    data_new.name = "l"
    
    #weight = th.ScaleCostWeight(1.0)
    cost = th.AutoDiffCostFunction(
        [g],
        loss_fn,
        batch_size * num_data,
        aux_vars=[data, data_new],
        name='pose_cost_fn')
    objective = th.Objective()
    objective.add(cost)
    #objective.update({'g': th.SO3(quaternion=torch.Tensor([0,0,0,1.0])),
    #                  'x': data})
    optimizer = th.GaussNewton(objective)
    layer = th.TheseusLayer(optimizer)

    values, info = layer.forward({
        'g': th.SO3(quaternion=torch.Tensor([0,0,0,1.0])),
        "x": data,
        'l': data_new
        #"w": w1
    }, optimizer_kwargs={"track_best_solution": True, "verbose":True})

    g = info.best_solution['g'].squeeze()
    #print("a @ iter {} is {}, loss is {}".format(i,a, loss))
    #print("a * data - data_rotated is {}".format(a @ data -data_rotated))
    print("g_init is {}".format(g_init))
    print("g is {}".format(g))
    print("gt is {}".format(gt))
    #print("gt * a.Inv() is {}".format(gt_a[0] @ a.Inv()))


def test_se3_constructor():
    a = pp.randn_SO3().requires_grad_(True)
    b = torch.Tensor([1,2,3]).requires_grad_(True)
    print("a.grad: ",a.grad)
    print(b.shape, a.tensor().shape)
    assert(a.is_leaf)
    assert(b.is_leaf)
    
    data = torch.randn((3,)).detach()

    T = pp.SE3(torch.cat((b, a.tensor()),dim=0))
    assert(T.requires_grad)
    assert(a.is_leaf)

    (T @ data).norm().backward()
    print("a.grad: ",a.grad)
    print("b.grad: ",b.grad)


class SE3Model(nn.Module):
    def __init__(self):
        super(SE3Model, self).__init__()

    def forward(self, T, target):
        return T @ target

def test_parallel_se3_R_t():
    batch_size = torch.cuda.device_count() * 800
    R = pp.identity_SO3().cuda().requires_grad_(True)
    t = torch.Tensor([1,2,3.0]).cuda().requires_grad_(True)
    T = pp.SE3(torch.cat((t, R.tensor()), dim=0))
    assert(T.requires_grad)
    T_init = T.clone().detach()

    gt_SE3 = pp.SE3([0,0,0,0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]).cuda().detach()
    gt_SE3_batch = gt_SE3.unsqueeze(0).expand(batch_size, gt_SE3.shape[0])

    data = torch.rand(batch_size, 3).cuda().detach() * 10    
    data_new =  (gt_SE3_batch @ data).detach()
        
    net = SE3Model()
    optimizer = torch.optim.Adam([R, t], lr=1e-4)
    for i in range(5000):
        T = pp.SE3(torch.cat((t, R.tensor()), dim=0))
        T_batch = T.unsqueeze(0).expand(batch_size, T_init.shape[0])
        result = net(T_batch, data)
        loss = (result - data_new).norm()
        loss.backward()
        optimizer.step()
        #print("R, t, T @ iter {} is {}, {}, {}, loss is {}".format(i,R, t, T, loss))
        print("R, t, T @ iter {} is {}, {}, loss is {}".format(i,R, t, loss))
        print("========================", T.grad)
        

    print("T_init is {}".format(T_init))
    print("T is {}".format(T))
    print("gt is {}".format(gt_SE3))
    print("gt * a.Inv() is {}".format(gt_SE3 @ T))


test_parallel_se3_R_t()   
#test_se3_constructor()
#test_parallel()

