# Author : Bryce Xu
# Time : 2019/12/12
# Function: Omniglot Data Loader

from DataSampler import OmniglotBatchSampler
from Dataset import OmniglotDataset
import numpy as np
import torch

def dataloader(parser, mode):
    dataset = OmniglotDataset(mode=mode, root=parser.dataset_root)
    classes_per_it = parser.classes_per_it_val
    num_samples = parser.num_support_val + parser.num_query_val
    sampler = OmniglotBatchSampler(labels=dataset.y, classes_per_it=classes_per_it, num_samples=num_samples,
                                       iterations=parser.iterations)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader