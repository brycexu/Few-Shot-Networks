# Author : Bryce Xu
# Time : 2019/12/17
# Function: 

import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os

class MiniImageNetDataset(data.Dataset):

    def __init__(self, mode, root):
        super(MiniImageNetDataset, self).__init__()
        csv_path = os.path.join(root, mode + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        self.data = []
        self.label = []
        self.wnids = []
        lb = -1
        for l in lines:
            name, wnid = l.split(',')
            path = os.path.join(root, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.label.append(lb)
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path, label = self.data[item], self.label[item]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label