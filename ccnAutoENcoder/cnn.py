import torch
import numpy as np
import scipy.io
from models import *
#model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)
model  = ConvAutoencoder()
model.load_state_dict(torch.load('weights.pth'))
for name, param in model.named_parameters():
    if param.requires_grad and name == 't_conv2.weight':
        t = param.data[0:27].detach().numpy()
        #print(t.shape)
        t =  np.reshape(t, (27, -1))
        #print(name)
        scipy.io.savemat('cnn3Filter.mat', {'mydata': t})

    