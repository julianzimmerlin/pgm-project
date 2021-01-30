import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from dataset import *
from models import *
from tqdm import tqdm
import scipy.io
transform = transforms.Compose([transforms.CenterCrop(32),
                               transforms.ToTensor()])


train_data = ImageLoader( dataset_dir = './BSR_bsds500/BSR/BSDS500/data/images/train/',transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=0)
model = ConvAutoencoder()
print(model)
'''
for name, param in model.named_parameters():
    if param.requires_grad:
        t = param.data[0:27].detach().numpy()
        print(t.shape)
        #t = 255 * np.reshape(t, (27, -1))
        print(name)
'''
#Loss function
criterion = nn.MSELoss()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
    
print(device)
model.to(device)
#Epochs
n_epochs = 600

for epoch in tqdm(range(1, n_epochs+1)):
    # monitor training loss
    train_loss = 0.0

    #Training
    for data in (train_loader):
        images = data
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)
          
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

torch.save(model.state_dict(), './weights.pth')
model  = model.to('cpu')
for name, param in model.named_parameters():
    if param.requires_grad:
        t = param.data.detach().numpy()
        print(t.shape)
        t = 255 * np.reshape(t, (27, -1))
        print(t.shape)
        scipy.io.savemat('cnnFilters.mat', {'mydata': t})

    break
