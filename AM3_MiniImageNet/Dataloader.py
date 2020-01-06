# Author : Bryce Xu
# Time : 2019/12/17
# Function: 

from DataSampler import MiniImageNetBatchSampler
from Dataset import MiniImageNetDataset
import torch
import numpy as np

def dataloader(parser, mode):
    dataset = MiniImageNetDataset(mode=mode, data_root=parser.dataset_root, label_root=parser.label_root,
                                  wmodel_root=parser.embedding_root)
    print(mode)
    print(len(dataset))
    if 'train' in mode:
        classes_per_it = parser.classes_per_it_tr
        num_samples = parser.num_support_tr + parser.num_query_tr
    else:
        classes_per_it = parser.classes_per_it_val
        num_samples = parser.num_support_val + parser.num_query_val
    sampler = MiniImageNetBatchSampler(labels=dataset.label, classes_per_it=classes_per_it, num_samples=num_samples,
                                   iterations=parser.iterations)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader
