# Author : Bryce Xu
# Time : 2019/11/18
# Function: 处理数据集

from Dataset_Utils import OmniglotDataset
from Dataset_Utils import PrototypicalBatchSampler
import numpy as np
import torch

def dataloader(parser, mode):
    dataset = OmniglotDataset(mode=mode, root=parser.dataset_root)
    print(mode)
    print(len(dataset))
    n_classes = len(np.unique(dataset.y))
    if n_classes < parser.classes_per_it_tr or n_classes < parser.classes_per_it_val:
        raise (Exception('There are not enough classes in the dataset in order ' +
                         'to satisfy the chosen classes_per_it. Decrease the ' +
                         'classes_per_it_{tr/val} option and try again.'))
    if 'train' in mode:
        classes_per_it = parser.classes_per_it_tr
        num_samples = parser.num_support_tr + parser.num_query_tr
    else:
        classes_per_it = parser.classes_per_it_val
        num_samples = parser.num_support_val + parser.num_query_val
    sampler = PrototypicalBatchSampler(labels=dataset.y, classes_per_it=classes_per_it, num_samples=num_samples, iterations=parser.iterations)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader
