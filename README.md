# STDA-inf
Style transfer for data augmentation: through in-data training and fusion classification.


## code structure
- run.sh: Run script, implemented with different mode (train or test) and different dataset (STL10 or CALTECH256 or CIFAR10).

- train.py: The main function of training.

- fusion_test.py: The main function of fusion test.

- dataset.py: Modified on the base of official Pytorch Implementation torchvision.datasets.ImageFolder, add style transfer module, and enlarge input path from one to many.

- utils.py: some utilized class or function: class of StyleTransform, function for Mixup augmentation, display bar, etc.

- models/: Models of various networks such as resnet, VGG. This project mainly adopts ResNet50. And there is a little difference between the ResNet50 different datasets. STL10, CALTECH256, and CIFAR10 correspond to resnet.py, resnet_caltech.py, and resnet_cifar.py respectively. ResNet50 for CALTECH256 is the same as ImageNet; ResNet50 for STL10 removes the first maxpool layer, changes the kernel_size of avgpool from 7 to 6; ResNet50 for CIFAR10 (refer to the Github implementation [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)) removes the first maxpool layer, changes the (kernel_size, stride, padding) of conv1 from (7, 2, 3) to (3, 1, 1), and changes the kernel_size of avgpool from 7 to 4.

- checkpoint/: Store checkpoints of trained models.

- log/: Store logs of training and test. 

- data/: 
  * cal_mean_std.py: Calculate the mean and std of dataset.
  * choose_style.py: Random choose in-data style images from the training dataset to save in the form of *list* or *dict*.
  * [STL10-data](https://cs.stanford.edu/~acoates/stl10/), [CALTECH256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/), [CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html) represent different dataset. As for the training and test dataset, you can download from the cloud disk [classification](https://disk.pku.edu.cn:443/link/F0B1ED091A1D5901B06358213A7CD533) with password `73f1`, unzip them and save as `train` `test` folder in the corresponding dataset path. CALTECH256 selects 60 images per class as training and remained images as test. CIFAR10 is downloaded from the official website and saved as images, so its class_to_idx may is not the same as torchvision.datasets.CIFAR10. The check_channel.py finds the 2-channel images of CALTECH256.
  
- extension:/
  * add_aug_train.py: Add some augmentation methods such as Mixup, Cutout, CutMix, and Manifold Mixup.
  * FGSM_test.py: FGSM robustness experiment.
  * cross_test.py: Test different test samples (original samples or style transferred samples) with different trained models (base training or out-data style training or in-data style training).
  * tSNE.py: Reduce dimension of dataset with tSNE method and visualize.

- AdaIN: Style transfer module, referred to the Github implementation [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN). 
  * list_transfer_interface.py: Interface of *list* mode of style transfer.
  * dict_transfer_interface.py: Interface of *dict* mode of style transfer.
  * `models` save trained models of AdaIN, you can download from the cloud disk [AdaIN-models](https://disk.pku.edu.cn:443/link/F212AA16F0ECC045040A457B28DC65DD) with password `EpHy`. Since the released pytorch version is 0.4.0 and our adopted is 1.1.0, we resave the model from *.t7 to *.pth (from torch to pytorch).
  * Other files are the same as the original released version with part removed.

## Start it
### Requirements


num_workers=0  if it's not 0, the code broadcast error, change it to 0.





