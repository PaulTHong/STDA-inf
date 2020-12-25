'''
Cross test with different trained models and test samples of different mode.
samples' mode: base or style-transferred (in or out or mixed)
model: base trained or in-data style trained or out-data style trained

Only implemented for STL-10.
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
import random
import sys

sys.path.insert(1, '/home/hongt/STDA-inf')
from models import *
from dataset import StyleDataset
from utils import StyleTransform
from AdaIN.list_transfer_interface import load_model

parser = argparse.ArgumentParser(description='PyTorch Cross Test')
parser.add_argument('--base_test_path', type=str, default='./data/STL10-data/test')
parser.add_argument('--ckpt_name', type=str, default='resnet50.pth')
parser.add_argument('--style_path', type=str, default='./data/STL10-data/stl_random_style_list_per10')
parser.add_argument('--content_size', type=int, default=0)
parser.add_argument('--style_size', type=int, default=0)
parser.add_argument('--mode', type=str, default='base', help='base or style')
parser.add_argument('--test_batch', type=int, default=1)  # 32
parser.add_argument('--seed', type=int, default=12345)
args = parser.parse_args()

print('\nArgparse space:') 
for k, v in args.__dict__.items():
    print(k, ': ', v)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vgg, decoder = load_model()  # load style transform model
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

print('==> Building Model..')
net = resnet.ResNet50()
if device == 'cuda':
    net = net.to(device)
    cudnn.benchmark = True

print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/' + args.ckpt_name)
net.load_state_dict(checkpoint['net'])


def test(mode):
    print('==> Preparing data..')
    data_path = args.base_test_path
    if mode == "base":
        testset = torchvision.datasets.ImageFolder(root=data_path, transform=transform_test)
    elif mode == "style":
        style_path = args.style_path
        print(style_path)
        style_transform = StyleTransform(style_path, vgg, decoder, transfer_prob=1.0, style_mode='list',
                                        content_size=args.content_size, style_size=args.style_size)
        testset = StyleDataset(root=data_path, transform=transform_test,
                               style_transform=style_transform, style_mode='list')

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=0)
    print('Test number: %d' % len(testset))

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return acc


def main():
    print('==> Testing..')
    mode = args.mode
    print('mode: ', mode)
    acc = test(mode)
    print('Test accuracy: %.4f\n' % acc)


if __name__ == '__main__':
    main()




