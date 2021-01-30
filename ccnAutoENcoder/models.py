import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv2d(3, 27, 3, padding=1,bias = False)
        #self.conv1.bias.data.fill_(0)  
        self.conv2 = nn.Conv2d(27, 4, 3, padding=1,bias = False)
        #self.conv2.bias.data.fill_(0)
        self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 27, 2, stride=2, bias = False)
        #self.t_conv1.bias.data.fill_(0)
        self.t_conv2 = nn.ConvTranspose2d(27, 3, 2, stride=2, bias = False)
        #self.t_conv2.bias.data.fill_(0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))
              
        return x