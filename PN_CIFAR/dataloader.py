import torchvision
from dataset import Dataset
from sampler import BatchSampler
from config import config
import torch

base_dir = '/home/ubuntu/project/'
data_dir = base_dir + 'cifar100/images/'
split_dir = base_dir + 'cifar100/splits/bertinetto/'

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_datasets = Dataset(data_dir=data_dir, split_dir=split_dir, mode="train", transform=train_transforms)
train_batch_sampler = BatchSampler(labels=train_datasets.labels, n_iter=config["n_iter"], n_way=config["n_way"], n_shot=config["n_shot"], n_query=config["n_query"])
train_loader = torch.utils.data.DataLoader(train_datasets, batch_sampler=train_batch_sampler, num_workers=4, pin_memory=True)

val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_datasets = Dataset(data_dir=data_dir, split_dir=split_dir, mode="val", transform=val_transforms)
val_batch_sampler = BatchSampler(labels=val_datasets.labels, n_iter=config["n_iter"], n_way=config["n_way"], n_shot=config["n_shot"], n_query=config["n_query"])
val_loader = torch.utils.data.DataLoader(val_datasets, batch_sampler=val_batch_sampler, num_workers=2, pin_memory=True)

test_datasets = Dataset(data_dir=data_dir, split_dir=split_dir, mode="test", transform=val_transforms)
test_batch_sampler = BatchSampler(labels=test_datasets.labels, n_iter=config["n_iter"], n_way=config["n_way"], n_shot=config["n_shot"], n_query=config["n_query"])
test_loader = torch.utils.data.DataLoader(test_datasets, batch_sampler=test_batch_sampler, num_workers=2, pin_memory=True)