from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from PIL import Image
import scipy.io
import torch

class FilterLoader(Dataset):
    def __init__(self, dataset_dir,transform = None):
        super(FilterLoader).__init__()
        self.filters =  scipy.io.loadmat(dataset_dir+'/filters.mat')['V']
        

        self.transform = transform
       
    def __getitem__(self, idx):
        single_filter = self.filters[:,idx].reshape(3,3,3)
        single_filter =  single_filter.astype('float32') 
        
        return torch.from_numpy(single_filter)
        
    def __len__(self):
        return self.filters.shape[1]