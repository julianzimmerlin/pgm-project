import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from PIL import Image
from functions import *
manualSeed = 999
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
img  = Image.open('check.jpg')
img = transforms.Resize(64)(img)
img = transforms.CenterCrop(64)(img)
noisyImgTensor = noisy(img,15)
print('Initial PSNR ', getPSNR(noisyImgTensor,transforms.ToTensor()(img)))
model = getModel(device)
T  = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(noisyImgTensor).to(device).detach().requires_grad_(True)
print('Is T leaf? {}'.format(T.is_leaf))
noisyImgTensor = noisyImgTensor.to(device)


for i in tqdm(range(10000)):
    T = optimize(T,noisyImgTensor,model,0.5,device)
    T = T.detach()
    #print(T.device)
    
    psnr = getPSNR(T.to('cpu'),transforms.ToTensor()(img))
    print('PSNR is {}'.format(psnr))







