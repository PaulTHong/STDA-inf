'''
Fusion test for classification.
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
import pickle
import cv2
from tqdm import tqdm
import numpy as np
import pickle
import sys

from models import *
from dataset import StyleDataset
from utils import StyleTransform
from AdaIN.list_transfer_interface import load_model

parser = argparse.ArgumentParser(description='PyTorch Test for Style Augmentation')
parser.add_argument('--dataset', type=str, default='STL10', choices=['STL10', 'CALTECH256', 'CIFAR10'],
                    help='CALTECH256 not recommended for dict test')
parser.add_argument('--ckpt_name', type=str, default='resnet50.pth')
parser.add_argument('--base_test_path', type=str, default='./data/STL10-data/test')
parser.add_argument('--include_base_test', action='store_true', help='also test in the original test dataset')
parser.add_argument('--style_path', type=str, default='./data/STL10-data/stl_random_style_list_per10',
                    help='style path')
parser.add_argument('--style_test_mode', type=str, default='list', choices=['list', 'dict', 'None'],
                    help='None means no fusion test')
parser.add_argument('--content_size', type=int, default=0, 
                    help='resize of content image during style transfer, 0 means no change')
parser.add_argument('--style_size', type=int, default=0, 
                    help='resize of style image during style transfer, 0 means no change')
parser.add_argument('--test_batch', type=int, default=1)  # 32
parser.add_argument('--transform_round', type=int, default=1)
parser.add_argument('--load_style_test', action='store_true', help='load the information of style test if true')
parser.add_argument('--info_path', type=str, default='./log/style_test_info')
args = parser.parse_args()

assert args.test_batch == 1, 'batch=1 is convenient to compute'
print('\nArgparse space:')
for k, v in args.__dict__.items():
    print(k, ': ', v) 

if not os.path.exists(args.info_path):
    os.mkdir(args.info_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = args.base_test_path
num_classes = {'STL10': 10, 'CIFAR10': 10, 'CALTECH256': 256}[args.dataset]
classes = torchvision.datasets.ImageFolder(root=args.base_test_path, transform=None).classes
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

print('==> Building Model..')
vgg, decoder = load_model()
if args.dataset == 'STL10':
    net = resnet.ResNet50(num_classes=num_classes)
elif args.dataset == 'CALTECH256':
    net = resnet_caltech.ResNet50(num_classes=num_classes)
elif args.dataset == 'CIFAR10':
    net = resnet_cifar.ResNet50(num_classes=num_classes)
else:
    raise NotImplementedError()
if device == 'cuda':
    net = net.to(device)
    cudnn.benchmark = True

print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
if device == 'cuda':
    checkpoint = torch.load('./checkpoint/' + args.ckpt_name)
elif device == 'cpu':
    checkpoint = torch.load('./checkpoint/' + args.ckpt_name, map_location='cpu')
net.load_state_dict(checkpoint['net'])


def test(style_class):
    '''
    style_class: the utilized style class in this round (e.g. 1~10 for STL-10),
    -1 means none style, 'all' means the styles in one round are from all classes.
    '''
    print('==> Preparing data..')
    if style_class == -1:  # base test
        testset = torchvision.datasets.ImageFolder(root=data_path, transform=transform_test)
    else:
        if style_class == 'all':  # list test
            style_path = args.style_path
        else:  # dict test
            style_path = os.path.join(args.style_path, str(style_class))
        print('Style path: ', style_path)
        style_transform = StyleTransform(style_path, vgg, decoder, transfer_prob=1.0, style_mode='list', 
                                         content_size=args.content_size, style_size=args.style_size, dataset=args.dataset)
        testset = StyleDataset(root=data_path, transform=transform_test,
                               style_transform=style_transform, style_mode='list')

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=0)  # 1
    test_num = len(testset)
    print('Test number: %d' % test_num)
    # classes = testset.classes
    # num_classes = len(classes)

    net.eval()
    correct = 0
    total = 0

    pred_vectors = torch.zeros((test_num, num_classes)).to(device)
    real_targets = torch.zeros(test_num).long().to(device)  # int type
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pred_vectors[batch_idx, :] = outputs
            real_targets[batch_idx] = targets
            
    acc = 100. * correct / total
    print('Single round accuracy: %.4f' % acc)
    return pred_vectors, real_targets


def main():
    if args.style_test_mode != 'None':
        print('==> Testing..')
        pre_round = 0

        info_name = os.path.split(args.ckpt_name)[-1][:-4]  # remove .pth
        info_path = os.path.join(args.info_path, args.dataset+'_'+info_name+'_style_test.info')
        if args.load_style_test:
            with open(info_path, 'rb') as f:
                info = pickle.load(f)
                avg_pred_vectors = info['avg_pred_vectors'].to(device)
                base_pred_vectors = info['base_pred_vectors'].to(device)
                pre_round = info['round']
                sum_pred_vectors = avg_pred_vectors * pre_round
        count = pre_round

        if args.style_test_mode == 'list':
            style_mode = ['all'] * (args.transform_round - pre_round)
        elif args.style_test_mode == 'dict':
            # When round is bigger than num_classes, replace the class id with the remainder of it divided by num_classes,
            # and replace 0 with num_classes. e.g. for STL-10, 11->1, ……, 19->9, 20->0->10, 21->1.
            style_mode = [cls % num_classes for cls in range(pre_round + 1, args.transform_round + 1)]
            style_mode = [num_classes if i == 0 else i for i in style_mode]
            style_mode = [classes[i] for i in style_mode]  # index to class name (name of subfolder)

        for style_class in style_mode:
            print('\n======== Round ' + str(count+1) + ' with style ' + str(style_class) + ' ========')
            count += 1
            if count == 1:
                sum_pred_vectors, real_targets = test(style_class)
                avg_pred_vectors = sum_pred_vectors
            else:
                pred_vectors, real_targets = test(style_class)
                sum_pred_vectors += pred_vectors
                avg_pred_vectors = sum_pred_vectors / count

            _, predicted = avg_pred_vectors.max(1)
            total = real_targets.size(0)
            correct = predicted.eq(real_targets).sum().item()
            acc = 100. * correct / total
            print('Until now test accuracy: %.4f' % acc)
        print('\nStyle transferred test accuracy: %.4f\n' % acc)

        assert args.include_base_test, "need to change a little about 'load and dump info' in code"
        if args.include_base_test:
            if not args.load_style_test:
                print('======== Base test ========')
                base_pred_vectors, real_targets = test(-1)
            # total = real_targets.size(0)

            # Grid search twice, the first time with step 0.1 and the second time with step 0.01.
            res = {}
            for alpha in [i * 0.1 for i in range(1, 10)]:
                final_pred_vectors = alpha * base_pred_vectors + (1 - alpha) * avg_pred_vectors
                _, predicted = final_pred_vectors.max(1)
                correct = predicted.eq(real_targets).sum().item()
                acc = 100. * correct / total
                res[alpha] = acc
                print('Alpha = %.2f, Final test accuracy: %.4f' % (alpha, acc))

            now_best = max(res.values())
            now_alpha = list(res.keys())[list(res.values()).index(now_best)]

            res = {}
            for i in range(-9, 10):
                alpha = now_alpha + 0.01 * i
                final_pred_vectors = alpha * base_pred_vectors + (1 - alpha) * avg_pred_vectors
                _, predicted = final_pred_vectors.max(1)
                correct = predicted.eq(real_targets).sum().item()
                acc = 100. * correct / total
                res[alpha] = acc
                print('Alpha = %.2f, Final test accuracy: %.4f' % (alpha, acc))
            now_best = max(res.values())
            now_alpha = list(res.keys())[list(res.values()).index(now_best)]
            print('\nBest alpha = %.2f, Best test accuracy: %.4f' % (now_alpha, now_best))

        with open(info_path, 'wb') as f:
            save_vectors = {'avg_pred_vectors': avg_pred_vectors,
                            'base_pred_vectors': base_pred_vectors,
                            'round': count}  # on GPU
            pickle.dump(save_vectors, f)
    else:  # only test for original test samples
        base_pred_vectors, real_targets = test(-1)
        total = real_targets.size(0)
        _, predicted = base_pred_vectors.max(1)
        correct = predicted.eq(real_targets).sum().item()
        acc = 100. * correct / total
        print('Base test accuracy: %.4f' % acc)


if __name__ == '__main__':
    main()




