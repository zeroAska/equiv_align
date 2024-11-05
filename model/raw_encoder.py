import torch.nn as nn

class IdentityEncoder(nn.Module):
    def __init__(self):
        super(IdentityEncoder, self).__init__()

    def forward(self, x):
        # x shape: [batch, rot_dim, num_points]
        return x.unsqueeze(1).detach(), x.unsqueeze(1).detach(), x.detach()
