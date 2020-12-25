"""
Train STL10 with PyTorch.
Implementation for some augmentation methods such as Mixup, Cutout, CutMix.
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

sys.path.insert(1, '/home/hongt/STDA-inf')
from dataset import StyleDataset
from models import *
from utils import StyleTransform, progress_bar, mixup_data, mixup_criterion, Cutout, rand_bbox
from AdaIN.list_transfer_interface import load_model

parser = argparse.ArgumentParser(description='PyTorch STL10 Training')
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
### Augmentation args:
parser.add_argument('--tra_augment', action='store_true', help='traditional augmentation including flip, crop, etc.')
# Style aug:
parser.add_argument('--no_style_aug', action='store_true', help='whether adopt style augmentation')
parser.add_argument('--style_path', type=str, default='./data/STL10-data/stl_random_style_list_per10',
                    help='style path')
parser.add_argument('--style_mode', type=str, default='list', choices=['single', 'list', 'dict'])
parser.add_argument('--transfer_prob', type=float, default=0.5)
parser.add_argument('--content_size', type=int, default=0)
parser.add_argument('--style_size', type=int, default=0)
# MixUp:
parser.add_argument('--mixup', action='store_true')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='mixup interpolation coefficient, also used in Manifold Mixup')
# CutOut:
parser.add_argument('--cutout', action='store_true', default=False, help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16, help='length of the holes')
# CutMix:
parser.add_argument('--cutmix', action='store_true')
parser.add_argument('--beta', default=0, type=float, help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float, help='cutmix probability')
# Manifold Mixup
parser.add_argument('--mixup_hidden', action='store_true')
parser.add_argument('--layer_mix', nargs='+', type=int, default=[0, 1, 2])
args = parser.parse_args()


def main():
    print('\nArgparse space:')
    for k, v in args.__dict__.items():
        print(k, ': ', v)
    # assert int(args.cutmix) + int(args.mixup) + int(args.cutout) == 1, 'only apply one augmentation one time'
    if args.train_num_workers > 1:
        import multiprocessing as mp
        mp.set_start_method('spawn')
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    if not args.no_style_aug:
        vgg, decoder = load_model()
        style_mode = args.style_mode
        style_transform = StyleTransform(args.style_path, vgg, decoder, transfer_prob=args.transfer_prob, style_mode=style_mode, 
                                         content_size=args.content_size, style_size=args.style_size, dataset=args.dataset)
    else:
        style_transform = None
        style_mode = None

    if args.tra_augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    if args.cutout:
        transform_train.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = StyleDataset(root=args.train_data_path, transform=transform_train,
                            style_transform=style_transform, style_mode=style_mode)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch,
                                              shuffle=True, num_workers=args.train_num_workers)
    print('Train number: %d' % len(trainset))
    testset = torchvision.datasets.ImageFolder(root=args.test_data_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Test number: %d' % len(testset))

    real_classes = ('airplane', 'truck', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship')
    print('Real classes: ', real_classes)
    classes = testset.classes
    print('ImageFolder classes:', classes)
    num_classes = len(classes)

    # Model
    print('==> Building model..')
    if args.mixup_hidden:
        net = manifold_resnet.ManifoldResNet50(num_classes=num_classes)
    else:
        net = resnet.ResNet50(num_classes=num_classes)
    # if device == 'cuda':
    if device != 'cpu':
        net = net.to(device)
        cudnn.benchmark = True

    best_acc = 0  # best test accuracy
    epoch_acc = {}
    start_epoch = 1  # start from epoch 1 or last checkpoint epoch
    if args.resume:
        # Load checkpoint
        ckpt_pth = './checkpoint/' + args.ckpt_name
        if os.path.isfile(ckpt_pth):
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(ckpt_path)
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
            if args.mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha)
                outputs = net(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            elif args.cutmix:
                cutmix_r = np.random.rand(1)
                if args.beta > 0 and cutmix_r < args.cutmix_prob:
                    # generate mixed sample
                    lam = np.random.beta(args.beta, args.beta)
                    rand_index = torch.randperm(inputs.size()[0]).to(device)
                    targets_a = targets
                    targets_b = targets[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                    # compute output
                    outputs = net(inputs)
                    loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (1. - lam)
                else:
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
            elif args.mixup_hidden:
                outputs, targets_a, targets_b, lam = net(inputs, targets, mixup_hidden=True, mixup_alpha=args.alpha,
                                                         layer_mix=args.layer_mix)  # layer_mix=None means [0,1,2]
                lam = lam[0]
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                """
                mixed_output, target_a, target_b, lam = model(inputs, targets, mixup_hidden=True,  mixup_alpha=args.alpha)
                output = net(inputs)
                lam = lam[0]
                target_a_one_hot = to_one_hot(target_a, args.num_classes)
                target_b_one_hot = to_one_hot(target_b, args.num_classes)
                mixed_target = target_a_one_hot * lam + target_b_one_hot * (1 - lam)
                loss = bce_loss(softmax(output), mixed_target)
                """
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if args.mixup or args.mixup_hidden or (args.cutmix and cutmix_r < args.cutmix_prob):
                correct += (lam * predicted.eq(targets_a).sum().item()
                            + (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        schedule.step()  # adjust learning rate

    # Test
    def test(epoch):
        nonlocal best_acc
        nonlocal epoch_acc
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

