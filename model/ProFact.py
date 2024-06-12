import torch
import torch.nn as nn
from torch.nn import functional as F

from .HolisticAttention import HA
from .CSPM import CSPM 
from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

     
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
    def forward(self, x):       
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class mlpHead_3(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(mlpHead_3, self).__init__()
        _, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*3,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, 1, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c2, c3, c4 = inputs

        ############## MLP decoder on C2-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c2.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c2.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1)) #256,64,64   256,64,64   

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
    
class mlpHead_4(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(mlpHead_4, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, 1, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
    
class ProFact(nn.Module):
    def __init__(self, ver = 'b0', pretrained = True):
        super(ProFact, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[ver]
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[ver](pretrained)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[ver]

        self.fem1 = CSPM(dim=self.in_channels[0], kernel_size=3)
        self.fem2 = CSPM(dim=self.in_channels[1], kernel_size=3)
        self.fem3 = CSPM(dim=self.in_channels[2], kernel_size=3)
        self.fem4 = CSPM(dim=self.in_channels[3], kernel_size=3)

        self.upsample05 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        self.HA = HA()

        self.decode_head_3 = mlpHead_3(self.in_channels, self.embedding_dim)
        self.decode_head_4 = mlpHead_4(self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        x = self.backbone.forward(inputs)
        x1, x2, x3, x4 = x 
        B = inputs.shape[0]
        

        x1_1 = self.fem1(x1)    
        x2_1 = self.fem2(x2)        
        x3_1 = self.fem3(x3)    
        x4_1 = self.fem4(x4)    
        y1 = x1_1, x2_1, x3_1, x4_1
        attention_map = self.decode_head_4(y1)   

        x2_2 = self.HA(self.upsample05(attention_map).sigmoid(), x2_1)   #x, 64, 64

        #----------------------------------#
        #   block5
        #----------------------------------#
        x3_2, H, W = self.backbone.patch_embed_5.forward(x2_2)
        for i, blk in enumerate(self.backbone.block_5):
            x3_2 = blk.forward(x3_2, H, W)
        x3_2 = self.backbone.norm_5(x3_2)
        x3_2 = x3_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        #----------------------------------#
        #   block6
        #----------------------------------#
        x4_2, H, W = self.backbone.patch_embed_6.forward(x3_2)
        for i, blk in enumerate(self.backbone.block_6):
            x4_2 = blk.forward(x4_2, H, W)
        x4_2 = self.backbone.norm_6(x4_2)
        x4_2 = x4_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        
        x2_2 = self.fem2(x2_2)
        x3_2 = self.fem3(x3_2)
        x4_2 = self.fem4(x4_2)
        y2 = x2_2, x3_2, x4_2

        detection_map = self.decode_head_3(y2)

        attention_map = F.interpolate(attention_map, size=inputs.shape[2:], mode='bilinear', align_corners=True)
        detection_map = F.interpolate(detection_map, size=inputs.shape[2:], mode='bilinear', align_corners=True)

        return attention_map, detection_map

