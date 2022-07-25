import numpy as np
import pydicom
import os
import torch
import scipy.io as sio
import scipy
import math
import matplotlib.pyplot as plt
from scipy.sparse.linalg import bicgstab
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class LData(torch.utils.data.Dataset):
    def __init__(self,**kwargs):
        super(LData, self).__init__()
        self.path=kwargs['datapath']
        self.train = kwargs['train']
        self.trainimgnum=kwargs['TrainImgNum']
        self.testimgnum=kwargs['TestImgNum']
        trainSetPath=[]
        testSetPath=[]
        if self.train :                
            for j in range(self.trainimgnum):
                traindir=os.path.join(self.path,"train/IMG")
                fname=traindir+str(j+1)+".mat"
                trainSetPath.append(fname)
            self.trainPath=(trainSetPath)
        else:
            for j in range(self.testimgnum):
                testdir=os.path.join(self.path,"test/IMG")
                fname=testdir+str(j+1)+".mat"
                testSetPath.append(fname)
            self.testPath=(testSetPath)
    def __getitem__(self, index):
        if self.train==True:
            imgpath=self.trainPath[index]
        else:
            imgpath=self.testPath[index]
        mat=sio.loadmat(imgpath)
        img=mat['Label']
        NoiseSino=mat['NoiseSino']
        fbpres=mat['FbpRes']
        return fbpres,NoiseSino,img

    def __len__(self):
        if self.train==True:
            return (self.trainimgnum) 
        else:
            return (self.testimgnum)
