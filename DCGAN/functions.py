import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
from PIL import Image
manualSeed = 999
from models import *
import math

def noisy(image,sigma):
    image = np.array(image)
    row,col,ch= image.shape
    mean = 0
    
    var = sigma * sigma 
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    print('noisy dtype is {}'.format(noisy.dtype))
    noisyImg = Image.fromarray(noisy.astype('uint8'))
    noisyImgTensor = transforms.ToTensor()(noisyImg)
    #noisyImgTensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(noisyImgTensor)
    noisyImgTensor = torch.unsqueeze(noisyImgTensor,0)
    return noisyImgTensor

def getModel(device):
    ndf = 64
    nc = 3
    ngpu = 1
    netD = Discriminator(ngpu,nc,ndf)
    netD.load_state_dict(torch.load('Discriminator_weights.pth'))
    netD = netD.to(device)
    return netD
    
    


    
def optimize(T,noisyImgTensor,model,sigma,device):
    lr = 0.008
    gradLlh = (noisyImgTensor-T)/sigma**2
    #print('T before grad descent {}'.format(T))
    optimizer = optim.Adam([T], lr=lr)
    optimizer.zero_grad()
    criterion = nn.BCELoss()
    model.zero_grad()
    #T = torch.unsqueeze(T,0)
    
    #print('T grad before optimize {}'.format(T.grad))
    prob = model(T).view(-1)
    label = torch.full((1,), 1, dtype=torch.float, device=device)
    logProb = criterion(prob, label)
    logProb.backward()
    #print('T grad after grad descent {}'.format(T.grad))
    optimizer.step()
   # print('T after grad descent {}'.format(T))
    #T = torch.squeeze(T,0)
    
    T = T + lr * gradLlh
    
    return T

def getPSNR(T,Original):
    T = np.array(T)
    Original = np.array(Original)
    mse = np.mean( (T - Original) ** 2 )
    PIXEL_MAX = 1   
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))