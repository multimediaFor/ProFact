import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

"""
Reference: 
https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CoTAttention.py
"""

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    

class CSPM(nn.Module):
    def __init__(self, dim=512, kernel_size=3, k=(1, 3, 5), e = 0.5): ##
        super(CSPM, self).__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )

        inter_channels = int(2 * dim * e)
        self.conv1 = Conv(dim, inter_channels, 1, 1)
        self.conv2 = Conv(inter_channels, inter_channels, 3, 1)
        self.conv3 = Conv(inter_channels, inter_channels, 1, 1)
        self.dilation = nn.ModuleList([nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=d, dilation=d) for d in k])
        self.conv4 = Conv(4 * inter_channels, inter_channels, 1, 1)
        self.conv_cat = Conv(2 * inter_channels, dim, 1, 1)
    
    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x) #bs,c,h,w
        v = self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y = torch.cat([k1,x],dim=1) #bs,2c,h,w
        att = self.attention_embed(y) #bs,c*k*k,h,w
        att = att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att = att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2 = F.softmax(att,dim=-1)*v
        k2 = k2.view(bs,c,h,w)

        k = k1+ k2

        x1 = self.conv3(self.conv2(self.conv1(k)))
        y1 = self.conv2(self.conv4(torch.cat([x1] + [d(x1) for d in self.dilation], 1)))
        y2 = self.conv1(k)

        return self.conv_cat(torch.cat((y1, y2), dim=1))

