'''
FGSM robustness. Only implemented for STL-10.
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
import sys
sys.path.insert(1, '/home/hongt/STDA-inf')
from models import *

parser = argparse.ArgumentParser(description='PyTorch FGSM Test')
parser.add_argument('--data_path', type=str, default='./data/STL10-data/test')
parser.add_argument('--ckpt_name', type=str, default='best_in_rand_list_per10_styleaug.pth')
parser.add_argument('--test_batch', type=int, default=1)
parser.add_argument('--epsilon', type=float, default=0.03, help='epsilon for FGSM and i-FGSM')
# parser.add_argument('--seed', type=int, default=1, help='random seed')
# parser.add_argument('--iteration', type=int, default=1, help='the number of iteration for FGSM')
# parser.add_argument('--alpha', type=float, default=2/255, help='alpha for i-FGSM')
args = parser.parse_args()

print('\nArgparse space:') 
for k, v in args.__dict__.items():
    print(k, ': ', v) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.ImageFolder(root=args.data_path, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=1)
print('Test number: %d' % len(testset))
classes = testset.classes
num_classes = len(classes)

print('==> Building Model..')
net = resnet.ResNet50()
if device == 'cuda':
    net = net.to(device)
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()

print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/' + args.ckpt_name)
net.load_state_dict(checkpoint['net'])


def test():
    net.eval()
    correct = 0
    total = 0
    correct_adv = 0

    if True:  # need grad information
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True  # Demand!
            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # print('Current accuracy: %.4f' % (100. * correct / total))

            # FGSM add disturbance
            x_grad = torch.sign(inputs.grad)
            x_adv = torch.clamp(inputs + args.epsilon * x_grad, -1, 1)
            outputs_adv = net(x_adv)

            _, predicted_adv = outputs_adv.max(1)
            correct_adv += predicted_adv.eq(targets).sum().item()

    acc = 100. * correct / total
    acc_adv = 100. * correct_adv / total
    print('Accuracy: %.4f' % acc)
    print('Adversarial accuracy: %.4f' % acc_adv)


if __name__ == '__main__':
    print('==> Testing..')
    test()


