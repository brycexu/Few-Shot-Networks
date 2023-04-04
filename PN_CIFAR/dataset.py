import torch
import os
import numpy as np
import PIL.Image as Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split_dir, mode, transform):
        self.data_dir = data_dir
        self.transform = transform
        split_file = os.path.join(split_dir, mode + '.csv')
        data = []
        labels = []
        with open(split_file, 'r') as f:
            for line in f.readlines()[1:]:
              if line.strip() != '':
                x, y = line.strip().split(',')
                data.append(x)
                labels.append(y)
        label_key = sorted(np.unique(np.array(labels)))
        label_map = dict(zip(label_key, range(len(label_key))))
        labels = [label_map[x] for x in labels]
        self.data = data
        self.labels = labels
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = Image.open(self.data_dir + self.data[index]).convert('RGB')
        label = self.labels[index]
        label = int(label)
        if self.transform:
            img = self.transform(img)
        return img, label