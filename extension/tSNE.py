"""
t-SNE: Visualize the distribution of STL-10.
"""
from time import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from PIL import Image
from sklearn.manifold import TSNE
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import sys

sys.path.insert(1, '/home/hongt/STDA-inf')
from models import *
from dataset import StyleDataset
from utils import StyleTransform
from AdaIN.list_transfer_interface import load_model

NUM_PER_CLASS = 30  # 100
SEED = 12345


def get_data():
    """ Base test samples. """
    data_path = '/home/hongt/STDA-inf/data/STL10-data/test'
    transform_test = None
    testset = torchvision.datasets.ImageFolder(root=data_path, transform=transform_test)  # In general, the dataset
    # is in the order of classes, so the following code about down sampling can be simplified further.
    total_len = len(testset)
    num_class = len(testset.classes)  # 10
    shape = np.array(testset[0][0]).shape  # 96x96x3 when transform is None
    print(np.array(testset[0][0]).shape)
    num_samples = num_class * NUM_PER_CLASS
    num_features = shape[0] * shape[1] * shape[2]

    samples = np.zeros((num_samples, num_features))
    labels = [0] * num_samples
    count = {}
    for i in range(num_class):
        count[i] = 0
    n_samp = 0
    for i in range(total_len):
        sample, label = testset[i]
        if count[label] < NUM_PER_CLASS:
            count[label] += 1
            samples[n_samp, :] = np.array(sample).flatten()
            labels[n_samp] = label
            n_samp += 1
    return samples, labels, num_samples, num_features


def get_style_data(style_mode='list'):
    """ Style transferred test samples."""
    data_path = '/home/hongt/Style_aug_released/data/STL10-data/test'
    if style_mode == 'list':
        style_path = '/home/hongt/STDA-inf/data/STL10-data/stl_random_style_list_per10'
    else:
        style_path = '/home/hongt/STDA-inf/data/STL10-data/stl_random_style_dict_per10'
    print('style_path: ', style_path)
    vgg, decoder = load_model()
    style_transform = StyleTransform(style_path, vgg, decoder, transfer_prob=1.0, style_mode=style_mode)
    transform_test = None

    testset = torchvision.datasets.ImageFolder(root=data_path, transform=transform_test)
    total_len = len(testset)
    num_class = len(testset.classes)  # 10
    class_to_idx = testset.class_to_idx
    shape = np.array(testset[0][0]).shape  # 96x96x3 when transform is None
    print(np.array(testset[0][0]).shape)
    num_samples = num_class * NUM_PER_CLASS
    num_features = shape[0] * shape[1] * shape[2]
    samples = np.zeros((num_samples, num_features))
    labels = [0] * num_samples
    count = {}
    for i in range(num_class):
        count[i] = 0
    n_samp = 0
    for i in tqdm(range(total_len)):
        sample, label = testset[i]
        if count[label] < NUM_PER_CLASS:
            count[label] += 1

            if style_transform is not None:
                if style_mode == 'list':
                    sample = style_transform(sample)
                elif style_mode == 'dict':
                    class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(label)]
                    sample = style_transform(sample, class_name)
                if not isinstance(sample, np.ndarray):
                    sample = np.array(sample)
                sample = sample.astype(np.uint8)
            if not isinstance(sample, Image.Image):
                sample = Image.fromarray(sample)
            if transform_test is not None:
                sample = transform_test(sample)

            samples[n_samp, :] = np.array(sample).flatten()
            # samples[n_samp, :] = sample.flatten()
            labels[n_samp] = label
            n_samp += 1
    return samples, labels, num_samples, num_features


total_feat_in = []
def hook_fn_forward(module, input, output):
    total_feat_in.append(input)


def get_featuremap_data():
    """ Feature maps of test samples. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = resnet.ResNet50()
    net = net.to(device)
    ckpt_path = './checkpoint/resnet50.pth'
    state_dict = torch.load(ckpt_path)
    net.load_state_dict(state_dict['net'])
    for name, layer in net.named_modules():
        if name == 'linear':  # hook the input of fc layer
            layer.register_forward_hook(hook_fn_forward)

    data_path = '/home/hongt/STDA-inf/data/STL10-data/test'
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(root=data_path, transform=None)
    total_len = len(testset)
    num_class = len(testset.classes)  # 10
    # shape = np.array(testset[0][0]).shape  # 96x96x3 when transform is None
    print(np.array(testset[0][0]).shape)
    num_samples = num_class * NUM_PER_CLASS
    num_features = 2048
    samples = np.zeros((num_samples, num_features))
    labels = [0] * num_samples
    count = {}
    for i in range(num_class):
        count[i] = 0
    n_samp = 0

    testset = torchvision.datasets.ImageFolder(root=data_path, transform=transform_test)
    with torch.no_grad():
        for i in tqdm(range(total_len)):
            sample, label = testset[i]
            sample = sample.to(device)
            if count[label] < NUM_PER_CLASS:
                count[label] += 1
                sample = sample.unsqueeze(0)
                net(sample)
                feature_map = total_feat_in[0][0].squeeze(0)
                # feature_map = net.last_feature_map.squeeze(0)
                # print(feature_map.size())
                samples[n_samp, :] = feature_map.cpu().numpy()  # .flatten() 
                labels[n_samp] = label
                n_samp += 1

                total_feat_in.clear()

    return samples, labels, num_samples, num_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    # === base test samples ===
    samples, labels, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=SEED, perplexity=30, n_iter=1000)
    # default: perplexity=30, n_iter=1000
    t0 = time()
    results = tsne.fit_transform(samples)
    plot_embedding(results, labels,
                         't-SNE embedding of STL-10: base test')
    print('time: %.2f' % (time() - t0))
    plt.savefig('base_test_t-SNE.png')
    
    
    # === style transferred test samples ===
    STYLE_MODE = 'dict'
    samples, labels, n_samples, n_features = get_style_data(style_mode=STYLE_MODE)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=SEED, perplexity=30, n_iter=1000)
    t0 = time()
    results = tsne.fit_transform(samples)
    plot_embedding(results, labels, 't-SNE embedding of STL-10: dict style transferred')
    print('time: %.2f' % (time() - t0))
    plt.savefig('dict_style_test_t-SNE.png')
    
    STYLE_MODE = 'list'
    samples, labels, n_samples, n_features = get_style_data(style_mode=STYLE_MODE)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=SEED, perplexity=30, n_iter=1000)
    t0 = time()
    results = tsne.fit_transform(samples)
    plot_embedding(results, labels, 't-SNE embedding of STL-10: list style transferred')
    print('time: %.2f' % (time() - t0))
    plt.savefig('list_style_test_t-SNE.png')


    # === feature maps of test samples ===
    samples, labels, n_samples, n_features = get_featuremap_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=SEED, perplexity=30, n_iter=1000)
    t0 = time()
    results = tsne.fit_transform(samples)
    plot_embedding(results, labels,
                   't-SNE embedding of STL-10\'s feature maps from fc layer')
    print('time: %.2f' % (time() - t0))
    plt.savefig('featuremap_test_t-SNE.png')


if __name__ == '__main__':
    main()

