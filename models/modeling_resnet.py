import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def np2th(weights,conv=False):
    weights = torch.from_numpy(weights)
    if conv:
        weights = weights.permute(3,2,0,1)
    return weights

def pjoin(*parts):
    key = "/".join(parts)
    return key


class StdConv2d(nn.Conv2d):
    def forward(self,x):
        w = self.weight
        v,m = torch.var_mean(w,dim=[1,2,3],keepdim=True,unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        inputs = F.conv2d(x,w, self.bias,self.stride,self.padding,self.dilation,self.groups)
        return inputs

def conv3x3(cin,cout,stride=1,groups=1,bias=False):
    conv = StdConv2d(cin,cout,kernel_size=3,stride=stride,padding=1,bias=bias,groups=groups)
    return conv

def conv1x1(cin,cout,stride=1,groups=1,bias=False):
    conv = StdConv2d(cin,cout,kernel_size=1,stride=stride,padding=0,bias=bias,groups=groups)
    return conv

class PreActBottleneck(nn.Module):
    def __init__(self,cin,cout=None,cmid=None,stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4
        self.conv1 = conv1x1(cin,cmid)
        self.gn1 = nn.GroupNorm(num_groups=32,num_channels=cmid,eps=1e-6,affine=True)
        self.conv2 = conv3x3(cin=cmid,cout=cmid,stride=stride)
        self.gn2 = nn.GroupNorm(num_groups=32,num_channels=cmid,eps=1e-6,affine=True)
        self.conv3 = conv1x1(cmid,cout)
        self.gn3 = nn.GroupNorm(num_groups=32,num_channels=cout,eps=1e-6,affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if (stride!=1 or cin!=cout):
            self.downsample = conv1x1(cin,cout,stride=stride)
            self.gn_proj = nn.GroupNorm(num_groups=32,num_channels=cout,eps=1e-6,affine=True)
    def forward(self,x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = self.gn_proj(residual)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.gn3(x)
        y = x+residual
        y = self.relu(y)
        return y
    def load_from(self,weights,n_block,n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)
        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])
        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])
        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])
        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)
        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))
        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))
        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))
        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])
            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    def __init__(self,width_factor,num_layers):
        super().__init__()
        block_units = num_layers
        self.width = int(64*width_factor)
        self.root = nn.Sequential(
            OrderedDict([
                ("conv",StdConv2d(3,self.width,kernel_size=7,stride=2,padding=3)),
                ("gn",nn.GroupNorm(32,self.width,eps=1e-6)),
                ("relu",nn.ReLU(inplace=True)),
                ("pool",nn.MaxPool2d(kernel_size=3,stride=2,padding=0))
            ])
        )
        self.body = nn.Sequential(
            OrderedDict([
                ("block1",nn.Sequential(
                    OrderedDict(
                        [("unit1",PreActBottleneck(cin=self.width,cout=self.width*4,cmid=self.width))]+
                        [(f"unit{i}",PreActBottleneck(cin=self.width*4,cout=self.width*4,cmid=self.width)) for i in range(2,block_units[0]+1)]
                    )
                )),
                ("block2",nn.Sequential(
                    OrderedDict(
                        [("unit1",PreActBottleneck(cin=self.width*4,cout=self.width*8,cmid=self.width*2,stride=2))]+
                        [(f"unit{i}",PreActBottleneck(cin=self.width*8,cout=self.width*8,cmid=self.width*2)) for i in range(2,block_units[1]+1)]
                    )
                )),
                ("block3",nn.Sequential(
                    OrderedDict(
                        [("unit1",PreActBottleneck(cin=self.width*8,cout=self.width*16,cmid=self.width*4,stride=2))]+
                        [(f"unit{i}",PreActBottleneck(cin=self.width*16,cout=self.width*16,cmid=self.width*4)) for i in range(2,block_units[2]+1)]
                    )
                ))
            ])
        )
    def forward(self,x):
        x = self.root(x)
        x = self.body(x)
        return x