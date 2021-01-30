from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from PIL import Image
class ImageLoader(Dataset):
    def __init__(self, dataset_dir,transform = None):
        super(ImageLoader).__init__()
        self.dir = dataset_dir
        self.file_list = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f))]

        self.transform = transform
       
    def __getitem__(self, idx):
        img =self.transform(Image.open(self.dir +  self.file_list[idx]))
        return img 
        
        
    def __len__(self):
        return len(self.file_list)