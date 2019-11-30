import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Net(nn.Module):
    def __init__(self, fc1_channel, fc2_channel):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, fc1_channel)
        self.fc2 = nn.Linear(fc1_channel, fc2_channel)
        self.fc3 = nn.Linear(fc2_channel, 10)
        self._weights_init()
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _weights_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)

def train_eval(train_dataloader, test_loader, model, optimizer,criterion, best_acc):
    # train process
    model.train()
    with torch.cuda.device(0):
        model.cuda()
        for idx, (data, label) in enumerate(train_dataloader):
            data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            predict = model(data)
            loss = criterion(predict, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # test process
    model.eval()
    accuracys = AverageMeter()
    with torch.no_grad():
        with torch.cuda.device(0):
            model.cuda()
            for idx, (data, label) in enumerate(test_loader):
                data = data.cuda()
                predict = model(data)
                accuray = calculate_accuracy(predict.data.cpu(), label)[0]
                accuracys.update(accuray)
    acc = accuracys.avg
    acc = acc.numpy()
    if acc > best_acc:
        best_acc = acc

    return model, optimizer, best_acc

def cal_acc(fc1_channel, fc2_channel):
    train_transform = transforms.Compose([
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
         ]) 

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    trainset = torchvision.datasets.CIFAR10(
                                    root='./data',
                                    train=True,
                                    download=False,
                                    transform=train_transform
                                )
    
    testset = torchvision.datasets.CIFAR10(
                                    root='./data',
                                    train=False,
                                    download=False,
                                    transform=test_transform
                                )
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    model = Net(fc1_channel, fc2_channel)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.8, last_epoch=-1)

    best_acc = 0
    # only train 2 epoch
    for epoch in range(2):
        model, optimizer, best_acc = train_eval(trainloader, testloader, model, optimizer, criterion, best_acc)
        scheduler.step()
    return best_acc

if __name__ == '__main__':
    import time
    start = time.time()
    acc = cal_acc(20,30)
    end = time.time()
    print('cost time --- {}'.format(end-start))
    print(type(acc))

