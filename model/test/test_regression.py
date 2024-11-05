
import torch
from torch import nn
import pypose as pp
import pypose.optim as ppos
from model.option import opt
from model.matcher.matcher import gen_matcher
from model.regression.regression import gen_regression

class SmallNet(nn.Module):
    def __init__(self, opt):
        super(SmallNet, self).__init__()
        self.is_train = (opt.exp_args.run_mode == 'train')
        
        self.matcher = gen_matcher(opt)
        self.regression = gen_regression(self.matcher, opt)

        #if self.is_train == False:
        ##    for param in self.encoder.parameters():
        #        param.requires_grad = False
                
    def forward(self, feat1, feat2, T_init):

        T_result = self.regression(feat1,feat2,
                                   feat1,
                                   feat2,
                                   T_init)
        
        return T_result




if __name__ =="__main__":


    torch.autograd.set_detect_anomaly(True)

    rand_x = torch.rand(size=(1, 4, 3, 20))


    rand_rot = pp.randn_SO3()
    
    rand_y = torch.permute(rand_rot @ torch.permute(rand_x, (0, 1, 3, 2)), (0, 1, 3, 2))
    
    small_net = SmallNet(opt)
    #torch.nn.init.xavier_uniform(small_net.parameters())

    idd = pp.identity_SO3()
    print(small_net(rand_x.cuda(), rand_y.cuda(), idd.cuda()))
    
    solver = ppos.solver.Cholesky()
    strategy = pp.optim.strategy.Adaptive(damping=1e-6)
    #self.matcher.set_T_grad_required(True)
    optimizer = pp.optim.LM(small_net,
                            solver=solver,
                            strategy=strategy,
                            min=1e-6,
                            vectorize=False)

    #iters = 0

    #while scheduler.continual():
    for iter in range(5):
        #pdb.set_trace()

        #residual = self.matcher(pc1, pc2, x, y, T)
        #residual.backward()
        residual = optimizer.step(input=(rand_x.cuda(), rand_y.cuda(),
                                         rand_x.cuda(), rand_y.cuda(), rand_rot), target=residual)
    
        #scheduler.step(residual)
        print('Residual is  %.7f @ %d it'%(residual, iters))
        #optimizer.zero_grad()
        #iters += 1
        #scheduler.step(residual)
        
    #xself.matcher.set_T_grad_required(False)

