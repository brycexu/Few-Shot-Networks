# Author : Bryce Xu
# Time : 2019/12/17
# Function: 

import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
from WordEmbedding import load_model, load_embeddings

class MiniImageNetDataset(data.Dataset):

    def __init__(self, mode, data_root, label_root, wmodel_root):
        super(MiniImageNetDataset, self).__init__()
        csv_path = os.path.join(data_root, mode + '.csv')
        data_lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        label_lines = [x.strip() for x in open(label_root, 'r').readlines()]
        self.data = []
        self.label = []
        self.wnids = []
        self.cld = {}
        self.label_name = []
        lb = -1
        self.word_model = load_model(wmodel_root)
        for l in label_lines:
            tl = l.split()
            if tl[0] not in self.cld.keys():
                self.cld[tl[0]] = tl[2].lower()
        for l in data_lines:
            name, wnid = l.split(',')
            path = os.path.join(data_root, 'images', name)
            label_name = self.cld[wnid]
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.label.append(lb)
            self.label_name.append(label_name)
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.embeddings = load_embeddings(self.label_name, self.word_model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path, label = self.data[item], self.label[item]
        image = self.transform(Image.open(path).convert('RGB'))
        label_embedding = self.embeddings[item]
        return image, label, label_embedding