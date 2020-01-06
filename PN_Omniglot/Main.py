# Author : Bryce Xu
# Time : 2019/11/18
# Function: 主函数

from Parser1 import get_parser1
from Parser2 import get_parser2
from Parser4 import get_parser3
from Parser3 import get_parser4
from Dataset import dataloader
from Network import Network
from Loss import loss_fn
import torch
import numpy as np
from Logger import Logger
import os

logger1 = Logger('./logs')
logger2 = Logger('./logs')
logger3 = Logger('./logs')
logger4 = Logger('./logs')

parser1 = get_parser1().parse_args()
parser2 = get_parser2().parse_args()
parser3 = get_parser3().parse_args()
parser4 = get_parser4().parse_args()

def main(parser, logger):
    print('--> Preparing Dataset:')
    trainset = dataloader(parser, 'train')
    valset = dataloader(parser, 'val')
    testset = dataloader(parser, 'test')
    print('--> Building Model:')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Network().to(device)
    print('--> Initializing Optimizer and Scheduler')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=parser.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                gamma=parser.lr_scheduler_gamma,
                                                step_size=parser.lr_scheduler_step)
    val_loss = []
    val_acc = []
    best_acc = 0
    best_model_path = os.path.join(parser.experiment_root, 'best_model.pth')
    best_state = model.state_dict()
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
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-parser.iterations:])
        avg_acc = 100. * np.mean(val_acc[-parser.iterations:])
        print('Validating Loss: {} | Accuracy: {}'.format(avg_loss, avg_acc))
        info = {'val_loss': avg_loss, 'val_accuracy': avg_acc}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)
        if avg_acc > best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
    # Testing
    model.load_state_dict(best_state)
    test_acc = []
    for epoch in range(10):
        for batch_index, (inputs, targets) in enumerate(testset):
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            _, acc = loss_fn(input=output, target=targets, n_support=parser.num_support_val)
            test_acc.append(acc.item())
    avg_acc = 100. * np.mean(test_acc)
    logger.scalar_summary('test_accuracy', avg_acc, 1)
    print('*****Testing Accuracy: {}'.format(avg_acc))

if __name__ == '__main__':
    print('--> Begin Trainin and Validating')
    print('5 way 5 shot')
    main(parser1, logger1)
    print('5 way 1 shot')
    main(parser2, logger2)
    print('20 way 5 shot')
    main(parser3, logger3)
    print('20 way 1 shot')
    main(parser4, logger4)

