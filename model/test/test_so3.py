
import pypose as pp
import pypose.optim as ppos
from torch import nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TestNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.nn_layer = nn.linear(3, 3)
        self.rotation = pp.Parameter(pp.randn_SO3(n))

    def forward(self, x):
        return self.rotation @ self.nn_layer(x)


n,epoch = 4, 5
net = TestNet(1).cuda()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.2, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4], gamma=0.5)

print("Before Optimization:\n", net.weight)
for i in range(epoch):
    optimizer.zero_grad()
    inputs = pp.randn_SO3(n).cuda()
    outputs = net(inputs)
    loss = outputs.abs().sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(loss)

print("Parameter:", count_parameters(net))
print("After Optimization:\n", net.weight)

