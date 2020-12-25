import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import random
import cv2
import numpy as np

import net
from function import adaptive_instance_normalization
from function import coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def get_path_list(root_dir):
    style_list = [os.path.join(root_dir, i) for i in os.listdir(root_dir)]
    style_list = [i for i in style_list if '.jpg' in i or '.png' in i or '.JPEG' in i]
    if len(style_list) == 0:
        raise Exception('Format of style images is not contained yet.')
    return style_list


def get_a_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class Object(dict):
    def __init__(self, *args, **kwargs):
        super(Object, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


args = Object()
dir_path = os.path.split(os.path.abspath(__file__))[0]
args.vgg = os.path.join(dir_path, 'models/vgg_normalised.pth')
args.decoder = os.path.join(dir_path, 'models/decoder.pth')
args.crop = False
args.alpha = 1.0
# args.save_ext = '.jpg'
# args.preserve_color = False
# args.style_interpolation_weights = ''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)
    return vgg, decoder


def list_style_transfer(img, style_path, vgg, decoder, content_size=0, style_size=0, dataset=None):

    '''
    img: content image, type: numpy.array()
    style_path: path list of style images
    '''
    # if dataset is None or dataset == 'STL10':
        # args.content_size = 0  # 0 means not resize during style transfer, default size of STL10 is 96
        # args.style_size = 0
    # elif dataset == 'CALTECH256':
        # args.content_size = 128  # 256
        # args.style_size = 128  # 256
    # elif dataset == 'CIFAR10':
        # args.content_size = 256  # 128
        # args.style_size = 256  # 128

    # print('content size: ', args.content_size)
    # print('style size: ', args.style_size)
    content_tf = test_transform(content_size, args.crop)
    style_tf = test_transform(style_size, args.crop)

    content = img
    (hh, ww) = np.array(content).shape[:2]

    style_list = style_path
    S = len(style_list)
    style_id = np.random.randint(0, S)  # random choice
    style_path = style_list[style_id]
    style = Image.open(style_path).convert('RGB')

    content = content_tf(content).unsqueeze(0)
    style = style_tf(style).unsqueeze(0)
    content = content.to(device)
    style = style.to(device)
    # vgg.to(device)
    # decoder.to(device)

    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style, args.alpha)

    output = output.cpu().data.float()
    output = (output * 255).clamp(0, 255).numpy()[0].transpose((1, 2, 0))

    # size of Caltech256 image is not the same, since it will be resized during training,
    # there is no need to repeat resizing it here.
    if dataset != 'CALTECH256':
        output = cv2.resize(output, (ww, hh), cv2.INTER_AREA)

    # print('Transfer finished!')
    return output
