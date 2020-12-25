# STDA-inf
Style transfer for data augmentation: through in-data training and fusion classification.

## code structure
run.sh: 
train.py: the main function of training.
fusion_test.py: the main function of fusion test.
dataset.py: modified on the base of official Pytorch Implementation torchvision.datasets.ImageFolder, add style transfer module, and enlarge input path from one to many.
utils.py: some utilized class or function, for example, class of StyleTransform, function for Mixup augmentation, display bar.


num_workers=0  if it's not 0, the code broadcast error, change it to 0.

extension/*.py  change the path

previous github addres:



data url of disk:
https://disk.pku.edu.cn:443/link/F0B1ED091A1D5901B06358213A7CD533
有效期限：2021-01-23 23:59
访问密码：73f1

