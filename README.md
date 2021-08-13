# [STDA-inf: Style Transfer for Data Augmentation Through In-data Training and Fusion Inference](https://link.springer.com/chapter/10.1007%2F978-3-030-84529-2_7) (ICIC 2021)
Official Implementation.

By *Tao Hong et al.*,
Department of Information and Computational Sciences, School of Mathematical Sciences and LMAM, Peking University, Beijing 100871, China

---
## Code structure
- `run.sh`: Run script, implemented with different modes (train or test) and different datasets (STL10 or CALTECH256 or CIFAR10).

- `train.py`: Main function of training.

- `fusion_test.py`: Main function of fusion test.

- `dataset.py`: Modified on the base of official Pytorch implementation `torch.utils.data.Dataset` and `torchvision.datasets.ImageFolder`. Add style transfer module into the data preprocess and enlarge the input path from single one to several (a list).

- `utils.py`: Some utilized class or function: class of StyleTransform, function for Mixup augmentation, display bar, etc.

- `models/`: Models of various networks such as resnet, VGG. This project mainly adopts `ResNet50`. And there is a little difference between the `ResNet50` for different datasets. STL10, CALTECH256, and CIFAR10 correspond to `resnet.py`, `resnet_caltech.py`, and `resnet_cifar.py` respectively. ResNet50 for CALTECH256 is the same as ImageNet; ResNet50 for STL10 removes the MaxPool layer after conv1, changes the kernel_size of AvgPool layer before fc from 7 to 6; ResNet50 for CIFAR10 (refer to the Github implementation [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)) removes the first MaxPool layer after conv1, changes the (kernel_size, stride, padding) of conv1 layer from (7, 2, 3) to (3, 1, 1), and changes the kernel_size of AvgPool layer before fc from 7 to 4. `manifold_resnet.py` is written for Manifold Mixup augmentation

- `checkpoint/`: Store checkpoints of trained models.

- `log/`: Store logs of training and test. Subfolder `style_test_info` stores the information of different rounds during fusion test.   

- `data/`: Store datasets.
  * `cal_mean_std.py`: Calculate the mean and std of dataset.
  * `choose_style.py`: Random choose in-data style images from the training dataset to save in the form of *list (intra-class)* or *dict (inter-class)*.
  * [STL10-data](https://cs.stanford.edu/~acoates/stl10/), [CALTECH256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/), [CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html) represent different datasets. As for the training and test datasets, you can download from the cloud disk [classification](https://disk.pku.edu.cn:443/link/850D3BE5F275167629B0D9B2FA71D098) with password `nGDZ`, unzip them and save as `train` / `test` subfolder in the corresponding data path. CALTECH256 selects 60 images per class as training and remained images as test. CIFAR10 is downloaded from the official website and saved as images and its `class_to_idx` is not the same as `torchvision.datasets.CIFAR10`. The `check_channel.py` finds the 2-channel images of CALTECH256.
  
- `extension/`: Some extended experiments.
  * `add_aug_train.py`: Add some augmentation methods such as Mixup, Cutout, CutMix, and Manifold Mixup.
  * `FGSM_test.py`: FGSM robustness experiment.
  * `cross_test.py`: Test different test samples (original samples or style transferred samples) with different trained models (base trained or out-data style trained or in-data style trained).
  * `tSNE.py`: Reduce the dimension of dataset with tSNE method and visualize its distribution.

- `AdaIN/`: Style transfer module, refer to the Github implementation [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN). 
  * `list_transfer_interface.py`: Interface of *list* mode of style transfer.
  * `dict_transfer_interface.py`: Interface of *dict* mode of style transfer.
  * `models/:` save trained models of AdaIN, you can download from the cloud disk [AdaIN-models](https://disk.pku.edu.cn:443/link/F212AA16F0ECC045040A457B28DC65DD) with password `EpHy`. Since the released pytorch version is 0.4.0 and our adopted version is 1.1.0, we resave the model from `*.t7` to `*.pth` (from torch to pytorch).
  * Other files are the same as the original released version with part removed.

## Start it
### Requirements
- `pytorch=1.1.0`
- `torchvison=0.3.0`
- `numpy`, `cv2`, `PIL`, `tqdm`, etc.

### Train
Take the STL10 dataset as example, train the **baseline** model with 2 gpus (the `GPU_DEVICE` assigns the gpu id and the `train_mode` chooses the train mode as `baseline` or `style_aug`): choose `train_mode` as `baseline` in `run.sh`, then run:
```
bash run.sh train STL10
```
which will execute the order:
```
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u train.py --dataset STL10 \
    --train_data_path ./data/STL10-data/train --test_data_path ./data/STL10-data/test \
    --ckpt_name STL10_baseline.pth --lr 0.001 --milestones 80 120 --gamma 0.1 --max_epoch 150 \
    --train_batch 256 --test_batch 256 --optim Adam --multi_gpu --train_num_workers 4 \
    --tra_augment --no_style_aug 2>&1 |tee log/STL10_baseline_train.log
```
---

Train the **in-data style-aug** model of STL10: choose `train_mode` as `style_aug` in `run.sh`, then run:
```
bash run.sh train STL10
```
which will execute the order:
```
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u train.py --dataset STL10 \
    --train_data_path ./data/STL10-data/train --test_data_path ./data/STL10-data/test \
    --ckpt_name STL10_in_rand_list_per10_styleaug.pth --lr 0.001 --milestones 80 120 --gamma 0.1 --max_epoch 150 \
    --train_batch 256 --test_batch 256 --optim Adam --multi_gpu --train_num_workers 0 \
    --style_path ./data/STL10-data/stl_random_style_list_per10 --transfer_prob 0.3 --style_mode list \
    --tra_augment 2>&1 |tee log/STL10_in_rand_list_per10_train.log
```

The training of CALTECH256 and CIFAR10 is similar (so not demonstrated in detail), which is executed as:
```
bash run.sh train CALTECH256
bash run.sh train CIFAR10
```

---
**Attention**:
Since there are two deep learning models in our proposed method, one for style transfer and one for classification, we need to take care of the correspondence between model, data and gpu id when adopting multi gpus. There is a parameter called `num_workers` in `torch.utils.data.DataLoader`, which represents the process number when loading data. The default value `0` means only one main process and several processes like `4` may be faster. However, if it broadcasts bugs like `CUDA Initialization Error` etc., you can change the `num_workers` of `trainloader` from `4` to `0`, then everything will go peacefully.

### Test
Still take the STL10 dataset as example, fusion test for 15 rounds with the trained in-data style-aug model, run:  
```
bash run.sh test STL10
```
which will execute the order:
```
for ((i=1; i<=1; i++)); do
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u fusion_test.py --base_test_path ./data/STL10-data/test \
    --include_base_test --style_test_mode list --style_path ./data/STL10-data/stl_random_style_list_per10 \
    --ckpt_name best_STL10_in_rand_list_per10_styleaug.pth --transform_round $i
     2>&1 |tee log/STL10_in_rand_list_per10_fusion_test.log
done
for ((i=2; i<=15; i++)); do
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u fusion_test.py --base_test_path ./data/STL10-data/test \
    --include_base_test --style_test_mode list --style_path ./data/STL10-data/stl_random_style_list_per10 \
    --ckpt_name best_STL10_in_rand_list_per10_styleaug.pth --transform_round $i --load_style_test
    2>&1 |tee -a log/STL10_in_rand_list_per10_fusion_test.log
done
```

### Others
As for the `extension/*.py`, you can run them in the path of `STDA-inf/`, but remember to change the path `sys.path.insert(1, [PATH])` in `*.py` to your own path. `extension/add_aug_train.py` calls `models/manifold_resnet.py`, which also needs to change the path. Global path is recommended (may avoid small bugs) though not superior in code portability.

---

If you have any questions, please contact `paul.ht@pku.edu.cn`.
