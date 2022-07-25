import numpy as np
import LoadData
from skimage.metrics import  peak_signal_noise_ratio as psnr
from SOUL_Net import nBlock
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import time
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--num_block", type=int, default=10)
parser.add_argument("--model_save_path", type=str, default="saved_models/1st")
parser.add_argument("--data_path", type=str, default="../../genedatas/ctlib_64_1024_72_58_98178_3.5_3/")#datas
parser.add_argument("--models_path", type=str, default="models/")
parser.add_argument("--testres_path", type=str, default="testres/")
parser.add_argument('--checkpoint_interval', type=int, default=1)
parser.add_argument('--initmethod',type=str,default="Fbp")
parser.add_argument('--method',type=str,default="SOUL_Net_ctlib_64_1024_72_58_98178_3.5_3_2ci")
parser.add_argument('--TrainImgNum',type=int, default=400)
parser.add_argument('--TestImgNum',type=int, default=100)
parser.add_argument('--Spectrumlen',type=int, default=5)
opt = parser.parse_args()
cuda = True if torch.cuda.is_available() else False
useCuda=cuda and True


def Train():
    if useCuda:
        print("use cuda")
        torch.cuda.set_device(0)
    else:
        print("Do not use cuda")
    LoadDatas= LoadData.LData(TrainImgNum=opt.TrainImgNum, TestImgNum=opt.TestImgNum, datapath=opt.data_path,  train=True)
    LoadTestDatas= LoadData.LData(TrainImgNum=opt.TrainImgNum, TestImgNum=opt.TestImgNum, datapath=opt.data_path, train=False)
    train_loader = DataLoader(dataset=LoadDatas, batch_size=opt.batch_size,shuffle=True)
    test_loader = DataLoader(dataset=LoadTestDatas, batch_size=opt.batch_size,shuffle=False)
    if useCuda:
        net=nBlock(blocknum=opt.num_block, views=64, dets=1024, width=256, height=256,
                   dImg=0.0072, dDet=0.0058, dAng=0.098178, s2r=3.5, d2r=3, binshift=0).cuda()
    else:
        net=nBlock(blocknum=opt.num_block,views=64, dets=1024, width=256, height=256,
                   dImg=0.0072, dDet=0.0058, dAng=0.098178, s2r=3.5, d2r=3, binshift=0)

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    l1criterion=torch.nn.L1Loss()
    Loss=[]
    for epoch in range(opt.epochs+1):
        starttime=time.time()
        traLoss=0
        for i,(x,Y,img) in enumerate(train_loader):
            if useCuda:
                batch_Sino=Variable(Y.float()).cuda()
                batch_x=Variable(x.float()).cuda()
            else:
                batch_Sino=Variable(Y.float())
                batch_x=Variable(x.float())
            batch_Label=Variable(img.float())
            net.train()
            if useCuda:
                out=net(batch_x,batch_Sino)
                out=out.cpu()
            else:
                out=net(batch_x,batch_Sino)
            finalLoss=l1criterion(out,batch_Label)
            optimizer.zero_grad()  
            torch.autograd.set_detect_anomaly(True)
            finalLoss.backward()

            optimizer.step() 
            traLoss+=finalLoss.data.item()
        print("epoch = ",epoch," loss = ",traLoss)
        Loss.append(traLoss)
        torch.save({ 'state_dict': net.state_dict(), 'Loss': Loss,#itnum,batchsize,lr
                     'optimizer': optimizer.state_dict()},
                   "../"+opt.models_path +opt.method+'_'+ str(opt.initmethod)+ '_'+str(opt.num_block)+ '_' + str(opt.batch_size) +'_'+str(opt.lr)+ '.pth')
        endtime=time.time()
        print('cost time= ',endtime-starttime,' s')
        totalSSIM=[]
        totalPSNR=[]
        if epoch%10==9 or epoch==0:
            SsimRes=[0]*5
            PredImg=[]
            PSNR = [0] * 5
            with torch.no_grad():
                net.eval()
                testloss=0
                for j,(testx,testY,testimg) in enumerate(test_loader):
                    testx=torch.Tensor(testx.float()).cuda()
                    testY=torch.Tensor(testY.float()).cuda()
                    testout=net(testx,testY)
                    testout=testout.cpu()
                    PredImg.append(testout)

                    for bat in range(opt.batch_size):
                        for spe in range(opt.Spectrumlen):
                            ssimtmp=ssim((testout[bat,spe,:,:].numpy()),(testimg[bat,spe,:,:].numpy()),data_range=1)
                            SsimRes[spe]+=ssimtmp

                            PSNRtmp = psnr((testout[bat, spe, :, :].numpy()), (testimg[bat, spe, :, :].numpy()),
                                           data_range=1)
                            PSNR[spe] += PSNRtmp
            SsimRes=np.array(SsimRes)
            SsimRes=SsimRes/(opt.TestImgNum)
            PSNR = np.array(PSNR)
            PSNR = PSNR / (opt.TestImgNum)
            totalSSIM.append(SsimRes)
            totalPSNR.append(PSNR)
            torch.save({ 'ssim': totalSSIM,
                         'psnr': totalPSNR,
                         'predimg': PredImg},
                       "../"+opt.testres_path +opt.method+'_'+ str(opt.initmethod)+ '_'+str(opt.num_block)+ '_' + str(opt.batch_size) + '_' + str(opt.lr)+ '.pth')
            print("test ssim= ",SsimRes)
            print("test PSNR= ", PSNR)

if __name__=="__main__":
    Train()



