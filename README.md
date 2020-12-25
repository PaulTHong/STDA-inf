# STDA-inf
Style transfer for data augmentation: through in-data training and fusion classification.

## code structure
-run.sh: Run script, implemented with different mode (train or test) and different dataset (STL10 or CALTECH256 or CIFAR10).

-train.py: The main function of training.

-fusion_test.py: The main function of fusion test.

-dataset.py: Modified on the base of official Pytorch Implementation torchvision.datasets.ImageFolder, add style transfer module, and enlarge input path from one to many.

-utils.py: some utilized class or function: class of StyleTransform, function for Mixup augmentation, display bar, etc.

-models/: Models of various networks such as resnet, VGG. This project mainly adopts ResNet50. And there is a little difference between the ResNet50 different datasets. STL10, CALTECH256, and CIFAR10 correspond to resnet.py, resnet_caltech.py, and resnet_cifar.py respectively. ResNet50 for CALTECH256 is the same as ImageNet; ResNet50 for STL10 removes the first maxpool layer, changes the kernel_size of avgpool from 7 to 6; ResNet50 for CIFAR10 (refer to the Github implementation [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)) removes the first maxpool layer, changes the (kernel_size, stride, padding) of conv1 from (7, 2, 3) to (3, 1, 1), and changes the kernel_size of avgpool from 7 to 4.

-checkpoint/: Store checkpoints of trained models.

-log/: Store logs of training and test. 

-data/:

-extension:/

-AdaIN: Refer to the Github implementation [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN). 



num_workers=0  if it's not 0, the code broadcast error, change it to 0.


previous github addres:



data url of disk:
https://disk.pku.edu.cn:443/link/F0B1ED091A1D5901B06358213A7CD533
有效期限：2021-01-23 23:59
访问密码：73f1

