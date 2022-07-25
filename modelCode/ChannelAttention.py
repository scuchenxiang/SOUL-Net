import ipdb
import torch.nn as nn
import torch
class channelAtten(nn.Module):
    def __init__(self,hidden=256):
        super(channelAtten, self).__init__()
        self.pool1=nn.AvgPool2d(256)
        self.seconv1=nn.Conv2d(hidden,hidden//16,kernel_size=1)
        self.seconv2=nn.Conv2d(hidden//16,hidden,kernel_size=1)
    def forward(self,x):
        b,c,h,w=x.shape

        avgpoolres=self.pool1(x)
        res=self.seconv1(avgpoolres)
        res=torch.relu(res)
        res=self.seconv2(res)
        res=torch.sigmoid(res)
        res=res*x
        # loss=torch.norm(res,1)/torch.norm(res,2)+torch.norm(res,1)
        return res