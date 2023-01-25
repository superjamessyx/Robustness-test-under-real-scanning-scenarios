import numpy as np
from torch.utils.data import Dataset
from PIL import Image
def default_loader(path):

    img = Image.open(path)
    return img.convert('RGB')

class customDataMultiScanner(Dataset):

    def __init__(self,  txt_path, dataset = '', data_transforms=None, loader = default_loader):
        self.img_name = np.load(txt_path).tolist()
        print("{} Mode: Contain {} images".format(dataset, len(self.img_name)))
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader
    def __len__(self):
        return len(self.img_name)
    def __getitem__(self, item):
        img_name = self.img_name[item]
        # label = self.img_label[item]
        img = self.loader(img_name)
        if self.data_transforms is not None:
            img = self.data_transforms(img)
        return img, img_name




class customDataDefaultScanner(Dataset):

    def __init__(self,  txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [line.strip().split('\t')[0] for line in lines]
            print("{} Mode: Contain {} images".format(dataset, len(self.img_name)))
            self.img_label = [int(line.strip().split('\t')[1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)
        img = self.data_transforms(img)
        return img, label, img_name
