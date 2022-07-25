import torch.nn as nn
import torch
from PrjModule import prj_module
from RDSVD import svdv2_1
from ChannelAttention import channelAtten

class Block(nn.Module):
    def __init__(self,options):
        super(Block, self).__init__()
        self.weight=nn.Parameter(torch.zeros(1))
        self.block1 = prj_module(self.weight,options)
        self.convmodel2 = nn.Sequential(
            nn.Conv2d(5,256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),

        )
        self.convmodel3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 5, 3, 1, 1),

        )
        self.thre = nn.Parameter(torch.zeros([1, 5]))
        self.thre1=nn.Parameter(torch.zeros(1))

        self.chanatten = channelAtten(256)
        self.rho=nn.Parameter(torch.zeros(1)+1)

    def func1(self,x):
        U,S,V=svdv2_1.apply(x)
        VT=V.permute(0,2,1)
        mythre = torch.sigmoid(self.thre1) * S[:,0]
        mythre=torch.unsqueeze(mythre,-1)
        S=S-mythre
        S=torch.relu(S)
        S = torch.diag_embed(S)
        US=torch.matmul(U,S)
        USV=torch.matmul(US,VT)
        return USV,0
    def RX(self,X):
        b,c,h,w=X.shape
        X_0=torch.reshape(X,[b,c,h*w])
        X_0=torch.permute(X_0,[0,2,1])
        return X_0
    def RTX(self,X,shape):
        b,c,h,w=shape

        X_0=torch.permute(X,[0,2,1])
        X_0 = torch.reshape(X_0, [b, c, h, w])
        return X_0
    def lowRankSparse(self,X,proj,BB):
        X_0=self.RX(X)
        X_1=X_0-BB
        Z,_ = self.func1(X_1)
        tmp=X_0-BB-Z

        temp=self.RTX(tmp,X.shape)
        r=self.fidelity(X, proj) -self.weight*self.rho*temp
        S_k=self.convmodel2(r)
        S_k=self.chanatten(S_k)
        S_k = self.convmodel3(S_k)
        S_k=S_k+r
        RXn=self.RX(S_k)
        BB=BB+Z-RXn
        return S_k,BB

    def fidelity(self,input,proj):
        b,c,wimg,himg=input.shape
        _,_, wsino, hsino = proj.shape
        tmp=torch.reshape(input,[b*c,1,wimg,himg])
        projtmp = torch.reshape(proj, [b * c, 1, wsino,hsino])
        tmp1=self.block1(tmp,projtmp)
        res = torch.reshape(tmp1, [b ,c,wimg,himg])
        return res


    def forward(self,myinput,proj,BB):
        res,BB=self.lowRankSparse(myinput,proj,BB)
        return res,BB
class nBlock(nn.Module):
    def __init__(self,**kwargs):
        super(nBlock,self).__init__()
        self.iternum=kwargs['blocknum']
        views = kwargs['views']
        dets = kwargs['dets']
        width = kwargs['width']
        height = kwargs['height']
        dImg = kwargs['dImg']
        dDet = kwargs['dDet']
        dAng = kwargs['dAng']
        s2r = kwargs['s2r']
        d2r = kwargs['d2r']
        binshift = kwargs['binshift']
        options = torch.Tensor([views, dets, width, height, dImg, dDet, dAng, s2r, d2r, binshift])
        Blocklist=[]
        for i in range(0, self.iternum):
            Blocklist.append(Block(options))
        self.Blocklist = nn.ModuleList(Blocklist)
    def forward(self,input1,proj):

        outputlist=[]
        outputlist.append(input1)
        b,c,h,w=input1.shape
        BB=torch.zeros([b,h*w,c]).cuda()
        for layer_idx in range(self.iternum):
            output,BB = self.Blocklist[layer_idx](outputlist[layer_idx],proj,BB)
            outputlist.append(output)
        return outputlist[-1]
