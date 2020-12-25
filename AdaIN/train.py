'''
Style Augmentation: Train ResNet50 with PyTorch.
On different datasets: STL-10, Caltech-256, CIFAR-10, etc.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import numpy as np
import sys

from dataset import StyleDataset
from models import resnet
from utils import StyleTransform, progress_bar
from AdaIN.list_transfer_interface import load_model

parser = argparse.ArgumentParser(description='PyTorch Training for Style Augmentation')
parser.add_argument('--dataset', type=str, default='STL10', help='STL10 or Caltech256 or CIFAR10')
parser.add_argument('--train_data_path', nargs='+', type=str, default=['./data/STL10-data/train/'],
                    help='path list is utilized in offline training of more than one data root')
parser.add_argument('--test_data_path', type=str, default='./data/STL10-data/test/')
parser.add_argument('--ckpt_name', type=str, default='resnet50.pth')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--milestones', nargs='+', type=int, default=[80, 120])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--max_epoch', default=150, type=int)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--train_batch', type=int, default=64)
parser.add_argument('--test_batch', type=int, default=64)
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--optim', type=str, default='Adam', help='Adam, SGD')
parser.add_argument('--train_num_workers', type=int, default=0,
                    help='need multiprocess when not 0, to avoid cuda error in style transfer of loading data')
# Augmentation args:
parser.add_argument('--tra_augment', action='store_true', help='traditional augmentation including flip, crop, etc.')
parser.add_argument('--no_style_aug', action='store_true', help='whether adopt style augmentation')
parser.add_argument('--style_path', type=str, default='./data/STL10-data/stl_random_style_list_per10',
                    help='style path')
parser.add_argument('--style_mode', type=str, default='list', help='single, list or dict')
parser.add_argument('--transfer_prob', type=float, default=0.5)
args = parser.parse_args()


def main():
    print('\nArgparse space:')
    for k, v in args.__dict__.items():
        print(k, ': ', v)

    if args.train_num_workers > 1:
        import multiprocessing as mp
        mp.set_start_method('spawn')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.current_device()

    # Data
    print('\n==> Preparing data..')
    if not args.no_style_aug:
        vgg, decoder = load_model()
        style_mode = args.style_mode
        style_transform = StyleTransform(args.style_path, vgg, decoder,
                                         transfer_prob=args.transfer_prob, style_mode=style_mode, dataset=args.data)
    else:
        style_transform = None
        style_mode = None

    if args.tra_augment:
        if args.data == 'STL10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(96, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.data == 'CALTECH256':
            raise NotImplementedError()
        elif args.data == 'CIFAR10':
            raise NotImplementedError()
        else:
            raise NotImplementedError('You can self-define the needed dataset.')
    else:
        if args.data == 'STL10':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.data == 'CALTECH256':
            raise NotImplementedError()
        elif args.data == 'CIFAR10':
            raise NotImplementedError()
        else:
            raise NotImplementedError('You can self-define the needed dataset.')

    if args.data == 'STL10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif args.data == 'CALTECH256':
        raise NotImplementedError()
    elif args.data == 'CIFAR10':
        raise NotImplementedError()
    else:
        raise NotImplementedError('You can self-define the needed dataset.')

    trainset = StyleDataset(root=args.train_data_path, transform=transform_train,
                            style_transform=style_transform, style_mode=style_mode)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch,
                                              shuffle=True, num_workers=args.train_num_workers)
    print('Train number: %d' % len(trainset))
    testset = torchvision.datasets.ImageFolder(root=args.test_data_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Test number: %d' % len(testset))

    classes = testset.classes  # class names
    print('ImageFolder classes:', classes)
    if args.data == 'STL10':
        real_classes = ('airplane', 'truck', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship')
        print('Real classes: ', real_classes)
    num_classes = len(classes)

    # Model
    print('\n==> Building model..')
    net = resnet.ResNet50(num_classes=num_classes)
    net = net.to(device)
    if device == 'cuda':
        net = net.cuda()
        cudnn.benchmark = True

    best_acc = 0  # best test accuracy
    epoch_acc = {}
    start_epoch = 1  # start from epoch 0 or the last checkpoint epoch
    if args.resume:
        # Load checkpoint.
        ckpt_pth = './checkpoint/' + args.ckpt_name
        if os.path.isfile(ckpt_pth):
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(ckpt_pth)
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch'] + 1

    if args.multi_gpu:
        net = torch.nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-4)
    else:
        raise NotImplementedError
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                    gamma=args.gamma, last_epoch=-1)
    # change 'last_epoch=start_epoch' will make bug
    if args.resume:
        for _ in range(start_epoch - 1):
            schedule.step()

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        print('lr: %.5f' % schedule.get_lr()[0])

        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        schedule.step()  # adjust learning rate

    # Test
    def test(epoch):
        global best_acc
        global epoch_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        epoch_acc[epoch] = acc
        if acc > best_acc:
            best_acc = acc
            save_best = True
        else:
            save_best = False
        state = {
            'net': net.module.state_dict() if args.multi_gpu else net.state_dict(),
            'best_acc': best_acc,
            # 'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.ckpt_name)
        if save_best:
            torch.save(state, './checkpoint/' + 'best_' + args.ckpt_name)
            print('Saving best acc: %.4f' % best_acc)

    start_time = time.time()
    for epoch in range(start_epoch, args.max_epoch + 1):
        train(epoch)
        test(epoch)
    end_time = time.time()
    print('Acc per epoch:')
    for e, a in epoch_acc.items():
        print('Epoch %d acc:%.4f' % (e, a))
    print('Best accuracy: %.4f' % best_acc)
    print('Time consume: %.2f s, i.e. %.2f h' % (end_time - start_time, (end_time - start_time) / 3600))





