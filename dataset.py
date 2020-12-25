'''
Modify based on pytorch source code. Inherit the class: torch.utils.data.Dataset
Add style transfer module into dataset.
And extend the dataset path from single to a list.
'''


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
from PIL import Image
import numpy as np


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class StyleDataset(Dataset):
    '''
    root: combine data from different dirs.
    '''
    def __init__(self, root, loader=default_loader, transform=None,
                 target_transform=None, style_transform=None, style_mode=None):
        assert len(root) > 0, 'No dataset root found!'
        if not isinstance(root, list):
            root = [root]
        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.style_transform = style_transform
        self.style_mode = style_mode
        self.samples = []
        for dir in root:
            subSet = datasets.ImageFolder(root=dir)
            self.class_to_idx = subSet.class_to_idx
            self.classes = subSet.classes
            print('Root: ' + dir + '; data number: ' + str(len(subSet.imgs)))
            self.samples.extend(subSet.imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)  # self.loader can be replaced by opencv2
        if self.style_transform is not None:
            if self.style_mode == 'list':
                sample = self.style_transform(sample)
            elif self.style_mode == 'dict':
                class_name = list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(target)]
                sample = self.style_transform(sample, class_name)

            if not isinstance(sample, np.ndarray):
                sample = np.array(sample)
            sample = sample.astype(np.uint8)  # if not np.uint8, ToTensor() won't divide 255!!!

        if not isinstance(sample, Image.Image):
            sample = Image.fromarray(sample)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


def main():
    path = ['./data/STL10-data/train/', './data/STL10-data/test/']
    # path = ['./data/CIFAR10/train/', './data/CIFAR10/test/']
    # path = ['./data/STL10-data/train/', './data/STL10-data/good_train/']
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = StyleDataset(root=path, transform=transform_test)
    # print(len(testset.samples))
    print(len(testset))
    # print('target: ', testset.__getitem__(0)[1], type(testset.__getitem__(0)[1]))
    print('target: ', testset[0][1], type(testset[0][1]))
    print(testset.samples[0])
    print(testset.classes)
    print(testset.class_to_idx)


if __name__ == '__main__':
    main()
