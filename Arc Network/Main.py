# Author : Bryce Xu
# Time : 2019/12/12
# Function: Main

from Parser1 import get_parser1
import torch
import torch.utils.data as data
from Logger import Logger
from Dataset import OmniglotDataset
import Network
import numpy as np
from Metric import ArcMarginProduct, eval
from torch.nn import DataParallel
from Dataloader import dataloader

logger1 = Logger('./logs')

parser1 = get_parser1().parse_args()

def main(parser, logger):
    print('--> Preparing Dataset:')
    trainset = OmniglotDataset(mode='train', root=parser.dataset_root)
    trainloader = data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
    valloader = dataloader(parser, 'val')
    #print(len(np.unique(trainset.y)))
    #print(len(trainset))
    #print(len(valloader))
    print('--> Building Model:')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Network.resnet18().to(device)
    model = DataParallel(model)
    metric = ArcMarginProduct(192, len(np.unique(trainset.y)), s=30, m=0.3).to(device)
    metric = DataParallel(metric)
    criterion = torch.nn.CrossEntropyLoss()
    print('--> Initializing Optimizer and Scheduler:')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=parser.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                gamma=parser.lr_scheduler_gamma,
                                                step_size=parser.lr_scheduler_step)
    for epoch in range(parser.epochs):
        print('\nEpoch: %d' % epoch)
        # Training
        train_loss = 0
        train_acc = 0
        train_correct = 0
        train_total = 0
        model.train()
        for batch_index, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device).long()
            feature = model(inputs)
            output = metric(feature, targets)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        scheduler.step()
        train_acc = 100.*train_correct / train_total
        print('Training Loss: {} | Accuracy: {}'.format(train_loss/100, train_acc))
        # Validating
        val_correct = 0
        val_total = 0
        model.eval()
        for batch_index, (inputs, targets) in enumerate(valloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            feature = model(inputs)
            correct = eval(input=feature, target=targets, n_support=parser.num_support_val)
            val_correct += correct
            val_total += 20
        val_acc = 100.*val_correct / val_total
        print('Accuracy: {}'.format(val_acc))
        info = {'train_accuracy': train_acc, 'val_accuracy': val_acc}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)

if __name__ == '__main__':
    print('--> Begin Trainin and Validating')
    main(parser1, logger1)