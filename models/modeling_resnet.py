import torch
import torch.nn as nn
import torch.nn.functional as F
def np2th(weights,conv=False):
    weights = torch.from_numpy(weights)
    if conv:
        weights = weights.permute(3,2,0,1)
    return weights

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