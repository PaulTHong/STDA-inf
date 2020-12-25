# STDA-inf
Style transfer for data augmentation: through in-data training and fusion classification.


## Code structure
- `run.sh`: Run script, implemented with different modes (train or test) and different datasets (STL10 or CALTECH256 or CIFAR10).

- `train.py`: Main function of training.

- `fusion_test.py`: Main function of fusion test.

- `dataset.py`: Modified on the base of official Pytorch implementation `torchvision.datasets.ImageFolder`. Add style transfer module and enlarge the input path from one to several.

- `utils.py`: Some utilized class or function: class of StyleTransform, function for Mixup augmentation, display bar, etc.

- `models/`: Models of various networks such as resnet, VGG. This project mainly adopts `ResNet50`. And there is a little difference between the `ResNet50` for different datasets. STL10, CALTECH256, and CIFAR10 correspond to `resnet.py`, `resnet_caltech.py`, and `resnet_cifar.py` respectively. ResNet50 for CALTECH256 is the same as ImageNet; ResNet50 for STL10 removes the first maxpool layer, changes the kernel_size of avgpool layer from 7 to 6; ResNet50 for CIFAR10 (refer to the Github implementation [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)) removes the first maxpool layer, changes the (kernel_size, stride, padding) of conv1 layer from (7, 2, 3) to (3, 1, 1), and changes the kernel_size of avgpool layer from 7 to 4.

- `checkpoint/`: Store checkpoints of trained models.

- `log/`: Store logs of training and test. Subfolder `style_test_info` stores the information of different rounds during fusion test.   

- `data/`: 
  * `cal_mean_std.py`: Calculate the mean and std of dataset.
  * `choose_style.py`: Random choose in-data style images from the training dataset to save in the form of *list* or *dict*.
  * [STL10-data](https://cs.stanford.edu/~acoates/stl10/), [CALTECH256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/), [CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html) represent different datasets. As for the training and test datasets, you can download from the cloud disk [classification](https://disk.pku.edu.cn:443/link/F0B1ED091A1D5901B06358213A7CD533) with password `73f1`, unzip them and save as `train` `test` subfolder in the corresponding data path. CALTECH256 selects 60 images per class as training and remained images as test. CIFAR10 is downloaded from the official website and saved as images and its class_to_idx is not the same as `torchvision.datasets.CIFAR10`. The `check_channel.py` finds the 2-channel images of CALTECH256.
  
- `extension/`:
  * `add_aug_train.py`: Add some augmentation methods such as Mixup, Cutout, CutMix, and Manifold Mixup.
  * `FGSM_test.py`: FGSM robustness experiment.
  * `cross_test.py`: Test different test samples (original samples or style transferred samples) with different trained models (base trained or out-data style trained or in-data style trained).
  * `tSNE.py`: Reduce dimension of dataset with tSNE method and visualize.

- `AdaIN/`: Style transfer module, refer to the Github implementation [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN). 
  * `list_transfer_interface.py`: Interface of *list* mode of style transfer.
  * `dict_transfer_interface.py`: Interface of *dict* mode of style transfer.
  * `models/` save trained models of AdaIN, you can download from the cloud disk [AdaIN-models](https://disk.pku.edu.cn:443/link/F212AA16F0ECC045040A457B28DC65DD) with password `EpHy`. Since the released pytorch version is 0.4.0 and our adopted version is 1.1.0, we resave the model from `*.t7` to `*.pth` (from torch to pytorch).
  * Other files are the same as the original released version with part removed.

## Start it
### Requirements


num_workers=0  if it's not 0, the code broadcast error, change it to 0.





