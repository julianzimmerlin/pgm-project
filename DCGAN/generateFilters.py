import torch
import numpy as np
import scipy.io
from models import *
#model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)
model  = Generator(1,100,64,3)
model.load_state_dict(torch.load('Generator_weights.pth'))
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
model.eval()
with torch.no_grad():
    model = model.to(device)
    noise = torch.randn(10, 100, 1, 1, device=device)
    tensorFilters = model(noise)
    numpyFilters =  tensorFilters.to('cpu').numpy()
    numpyFilters = numpyFilters.reshape(27,-1)
    scipy.io.savemat('ganFilters.mat', {'ganFilters': numpyFilters})

    