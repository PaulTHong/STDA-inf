"""
Style Augmentation: Train ResNet50 with PyTorch.
On different datasets: STL-10, CALTECH-256, CIFAR-10, etc.
"""
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
from models import *
from utils import StyleTransform, progress_bar
from AdaIN.list_transfer_interface import load_model

parser = argparse.ArgumentParser(description='PyTorch Training for Style Augmentation')
parser.add_argument('--dataset', type=str, default='STL10', choices=['STL10', 'CALTECH256', 'CIFAR10'])
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
parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
parser.add_argument('--train_num_workers', type=int, default=0,
                    help='need multiprocess when not 0, to avoid cuda error in style transfer of loading data')
# Augmentation args:
parser.add_argument('--tra_augment', action='store_true', help='traditional augmentation including flip, crop, etc.')
parser.add_argument('--no_style_aug', action='store_true', help='whether adopt style augmentation')
parser.add_argument('--style_path', type=str, default='./data/STL10-data/stl_random_style_list_per10',
                    help='style path')
parser.add_argument('--style_mode', type=str, default='list', choices=['single', 'list', 'dict'])
parser.add_argument('--transfer_prob', type=float, default=0.5)
parser.add_argument('--content_size', type=int, default=0, 
                    help='resize of content image during style transfer, 0 means no change')
parser.add_argument('--style_size', type=int, default=0, 
                    help='resize of style image during style transfer, 0 means no change')
args = parser.parse_args()


def main():
    print('\nArgparse space:')
    for k, v in args.__dict__.items():
        print(k, ': ', v)

    if args.train_num_workers > 1:
        import multiprocessing as mp
        mp.set_start_method('spawn')

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # Data
    print('\n==> Preparing data..')
    if not args.no_style_aug:
        vgg, decoder = load_model()
        style_mode = args.style_mode
        style_transform = StyleTransform(args.style_path, vgg, decoder, args.transfer_prob, style_mode, 
                                         args.content_size, args.style_size, args.dataset)
    else:
        style_transform = None
        style_mode = None

    if args.tra_augment:
        if args.dataset == 'STL10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(96, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.dataset == 'CALTECH256':
            transform_train = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # mean and std of ImageNet
            ])
        elif args.dataset == 'CIFAR10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            raise NotImplementedError('You can customize the required dataset.')
    else:
        if args.dataset == 'STL10':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.dataset == 'CALTECH256':
            transform_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        elif args.dataset == 'CIFAR10':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            raise NotImplementedError('You can customize the required dataset.')

    if args.dataset == 'STL10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif args.dataset == 'CALTECH256':
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif args.dataset == 'CIFAR10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise NotImplementedError('You can customize the required dataset.')

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
    if args.dataset == 'STL10':
        real_classes = ('airplane', 'truck', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship')
        print('Real classes: ', real_classes)
    # The default class_to_idx of torchvision.datasets.CIFAR10 is:
    # {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    # 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    # While our processed is: {'bird': 0, 'car': 1, 'cat': 2, 'deer': 3, 'dog': 4,
    # 'frog': 5, 'horse': 6, 'plane': 7, 'ship': 8, 'truck': 9}
    num_classes = len(classes)

    # Model
    print('\n==> Building model..')
    if args.dataset == 'STL10':
        net = resnet.ResNet50(num_classes=num_classes)
    elif args.dataset == 'CALTECH256':
        net = resnet_caltech.ResNet50(num_classes=num_classes)
    elif args.dataset == 'CIFAR10':
        net = resnet_cifar.ResNet50(num_classes=num_classes)
    else:
        raise NotImplementedError()

    # if device == 'cuda':
    if device != 'cpu':
        net = net.to(device)
        cudnn.benchmark = True

    best_acc = 0  # best test accuracy
    epoch_acc = {}
    start_epoch = 1  # start from epoch 1 or the last checkpoint epoch
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
        nonlocal best_acc
        nonlocal epoch_acc  # 'global' will make error since epoch_acc is still in function 'main', not global variable
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
    print('\nAcc per epoch:')
    for e, a in epoch_acc.items():
        print('Epoch %d acc:%.4f' % (e, a))
    print('Best accuracy: %.4f' % best_acc)
    print('Time consume: %.2f s, i.e. %.2f h' % (end_time - start_time, (end_time - start_time) / 3600))


if __name__ == '__main__':
    main()


