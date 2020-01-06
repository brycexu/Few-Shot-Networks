# Author : Bryce Xu
# Time : 2019/11/18
# Function: 主函数

from Parser1 import get_parser1
from Parser3 import get_parser3
from Parser5 import get_parser5
from Parser10 import get_parser10
from Parser15 import get_parser15
from Dataset import dataloader, dataloader2
from Network import Network
from Loss import loss_fn
import torch
import numpy as np
from Logger import Logger
import os

logger1 = Logger('./logs')
logger3 = Logger('./logs')
logger5 = Logger('./logs')
logger10 = Logger('./logs')
logger15 = Logger('./logs')

parser1 = get_parser1().parse_args()
parser3 = get_parser3().parse_args()
parser5 = get_parser5().parse_args()
parser10 = get_parser10().parse_args()
parser15 = get_parser15().parse_args()

def main(parser, logger):
    print('--> Preparing Dataset:')
    trainset = dataloader(parser, 'train')
    valset = dataloader(parser, 'val')
    valset2 = dataloader2(parser, 'val')
    testset = dataloader(parser, 'test')
    testset2 = dataloader2(parser, 'test')
    print('--> Building Model:')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Network().to(device)
    print('--> Initializing Optimizer and Scheduler')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=parser.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                gamma=parser.lr_scheduler_gamma,
                                                step_size=parser.lr_scheduler_step)
    val_loss1 = []
    val_acc1 = []
    best_acc1 = 0
    best_state1 = model.state_dict()
    val_loss2 = []
    val_acc2 = []
    best_acc2 = 0
    best_state2 = model.state_dict()
    for epoch in range(parser.epochs):
        print('\nEpoch: %d' % epoch)
        # Training
        model.train()
        for batch_index, (inputs, targets) in enumerate(trainset):
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss, acc = loss_fn(input=output, target=targets, n_support=parser.num_support_tr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        # Validating
        model.eval()
        for batch_index, (inputs, targets) in enumerate(valset):
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss, acc = loss_fn(input=output, target=targets, n_support=parser.num_support_val)
            val_loss1.append(loss.item())
            val_acc1.append(acc.item())
        avg_loss = np.mean(val_loss1[-parser.iterations:])
        avg_acc = 100. * np.mean(val_acc1[-parser.iterations:])
        print('Validating1 Loss: {} | Accuracy1: {}'.format(avg_loss, avg_acc))
        info = {'val_loss1': avg_loss, 'val_accuracy1': avg_acc}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)
        if avg_acc > best_acc1:
            best_acc1 = avg_acc
            best_state1 = model.state_dict()
        for batch_index, (inputs, targets) in enumerate(valset2):
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss, acc = loss_fn(input=output, target=targets, n_support=parser.num_support_val2)
            val_loss2.append(loss.item())
            val_acc2.append(acc.item())
        avg_loss = np.mean(val_loss2[-parser.iterations:])
        avg_acc = 100. * np.mean(val_acc2[-parser.iterations:])
        print('Validating2 Loss: {} | Accuracy2: {}'.format(avg_loss, avg_acc))
        info = {'val_loss2': avg_loss, 'val_accuracy2': avg_acc}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)
        if avg_acc > best_acc2:
            best_acc2 = avg_acc
            best_state2 = model.state_dict()
    # Testing
    model.load_state_dict(best_state1)
    test_acc = []
    for epoch in range(10):
        for batch_index, (inputs, targets) in enumerate(testset):
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            _, acc = loss_fn(input=output, target=targets, n_support=parser.num_support_val)
            test_acc.append(acc.item())
    avg_acc = 100. * np.mean(test_acc)
    logger.scalar_summary('test_accuracy1', avg_acc, 1)
    print('*****Testing1 Accuracy: {}'.format(avg_acc))
    model.load_state_dict(best_state2)
    test_acc2 = []
    for epoch in range(10):
        for batch_index, (inputs, targets) in enumerate(testset2):
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            _, acc = loss_fn(input=output, target=targets, n_support=parser.num_support_val2)
            test_acc2.append(acc.item())
    avg_acc = 100. * np.mean(test_acc2)
    logger.scalar_summary('test_accuracy2', avg_acc, 1)
    print('*****Testing2 Accuracy: {}'.format(avg_acc))

if __name__ == '__main__':
    print('--> Begin Trainin and Validating')
    #print('NS 1:')
    #main(parser1, logger1)
    #print('NS 3:')
    #main(parser3, logger3)
    #print('NS 5:')
    #main(parser5, logger5)
    #print('NS 10:')
    #main(parser10, logger10)
    print('NS 15:')
    main(parser15, logger15)

