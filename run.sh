# !/bin/bash
# $1 as 'train' or 'test' mode
# $2 as dataset, 'STL10' or 'CALTECH256' or 'CIFAR10'

GPU_NUM=2
GPU_DEVICE=0,1  # 2 gpus for train and 1 gpu for test. 1 gpu for CIFAR10.

#train_mode="baseline"
train_mode="style_aug"

if [ $1 = "train" ]; then
  echo "Train mode:"
  echo $train_mode
  case $2 in
  STL10)
    if [ $train_mode = "baseline" ]; then
        #srun --job-name=stl_train -p GPU36 --gres=gpu:2 --qos low --time 120:00:00 \
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u train.py --ckpt_name STL10_baseline.pth --optim Adam \
            --train_batch 256 --test_batch 256 --tra_augment --no_style_aug --multi_gpu --train_num_workers 4 \
            2>&1 |tee log/STL10_baseline_train.log

        #CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u train.py --dataset STL10 \
            #--train_data_path ./data/STL10-data/train --test_data_path ./data/STL10-data/test \
            #--ckpt_name STL10_baseline.pth --lr 0.001 --milestones 80 120 --gamma 0.1 --max_epoch 150 \
            #--train_batch 256 --test_batch 256 --optim Adam --multi_gpu --train_num_workers 4 \
            #--tra_augment --no_style_aug 2>&1 |tee log/STL10_baseline_train.log
    elif [ $train_mode = "style_aug" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u train.py --dataset STL10 \
            --train_data_path ./data/STL10-data/train --test_data_path ./data/STL10-data/test \
            --ckpt_name STL10_in_rand_list_per10_styleaug.pth --lr 0.001 --milestones 80 120 --gamma 0.1 --max_epoch 150 \
            --train_batch 256 --test_batch 256 --optim Adam --multi_gpu --train_num_workers 0 \
            --style_path ./data/STL10-data/stl_random_style_list_per10 --transfer_prob 0.3 --style_mode list \
            --tra_augment 2>&1 |tee log/STL10_in_rand_list_per10_train.log

        # ====== out-data style ======:
        #CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u train.py --dataset STL10 \
            #--train_data_path ./data/STL10-data/train --test_data_path ./data/STL10-data/test \
            #--ckpt_name STL10_out_rand_list_10_styleaug.pth --lr 0.001 --milestones 80 120 --gamma 0.1 --max_epoch 150 \
            #--train_batch 256 --test_batch 256 --optim Adam --multi_gpu --train_num_workers 0 \
            #--style_path ./data/out_random_style_list_10 --transfer_prob 0.3 --style_mode list --content_size 0 --style_size 96 \
            #--tra_augment 2>&1 |tee log/STL10_out_rand_list_10_train.log
    fi
  ;;
  CALTECH256)
    if [ $train_mode = "baseline" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u train.py --dataset CALTECH256 \
            --train_data_path ./data/CALTECH256/train --test_data_path ./data/CALTECH256/test \
            --ckpt_name CALTECH256_baseline.pth --lr 0.01 --milestones 60 120 --gamma 0.2 --max_epoch 150 \
            --train_batch 32 --test_batch 32 --optim SGD --train_num_workers 4 \
            --tra_augment --no_style_aug --multi_gpu 2>&1 |tee log/CALTECH256_baseline_train.log
    elif [ $train_mode = "style_aug" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u train.py --dataset CALTECH256 \
            --train_data_path ./data/CALTECH256/train --test_data_path ./data/CALTECH256/test \
            --ckpt_name CALTECH256_in_rand_list_100_styleaug.pth --lr 0.01 --milestones 60 120 --gamma 0.2 --max_epoch 150 \
            --train_batch 32 --test_batch 32 --optim SGD --train_num_workers 0 --content_size 128 --style_size 128 \
            --style_path ./data/CALTECH256/cal_random_style_list_100 --transfer_prob 0.3 --style_mode list \
            --tra_augment --multi_gpu 2>&1 |tee log/CALTECH256_in_rand_list_100_train.log
    fi
  ;;
  CIFAR10)
    if [ $train_mode = "baseline" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u train.py --dataset CIFAR10 \
            --train_data_path ./data/CIFAR10/train --test_data_path ./data/CIFAR10/test \
            --ckpt_name CIFAR10_baseline.pth --lr 0.1 --milestones 60 120 160 --gamma 0.2 --max_epoch 200 \
            --train_batch 128 --test_batch 128 --optim SGD --train_num_workers 4 \
            --tra_augment --no_style_aug 2>&1 |tee log/CIFAR10_baseline_train.log
    elif [ $train_mode = "style_aug" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u train.py --dataset CIFAR10 \
            --train_data_path ./data/CIFAR10/train --test_data_path ./data/CIFAR10/test \
            --ckpt_name CIFAR10_in_rand_list_per10_styleaug.pth --lr 0.1 --milestones 60 120 160 --gamma 0.2 --max_epoch 200 \
            --train_batch 128 --test_batch 128 --optim SGD --train_num_workers 0 --content_size 256 --style_size 256 \
            --style_path ./data/CIFAR10/in_random_style_list_per10 --transfer_prob 0.3 --style_mode list \
            --tra_augment 2>&1 |tee log/CIFAR10_in_rand_list_per10_train.log
    fi
  ;;
  esac
elif [ $1 = "test" ]; then
  case $2 in
  STL10)
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
  ;;
  CALTECH256)
    for ((i=1; i<=1; i++)); do
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u fusion_test.py --base_test_path ./data/CALTECH256/test \
        --include_base_test --style_test_mode list --style_path ./data/CALTECH256/cal_random_style_list_100 \
        --ckpt_name best_CALTECH256_in_rand_list_100_styleaug.pth --transform_round $i --content_size 128 --style_size 128
    done
    for ((i=2; i<=15; i++)); do
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u fusion_test.py --base_test_path ./data/CALTECH256/test \
        --include_base_test --style_test_mode list --style_path ./data/CALTECH256/cal_random_style_list_100 \
        --ckpt_name best_CALTECH256_in_rand_list_100_styleaug.pth --transform_round $i \
         --content_size 128 --style_size 128 --load_style_test
    done
  ;;
  CIFAR10)
    for ((i=1; i<=1; i++)); do
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u fusion_test.py --base_test_path ./data/CIFAR10/test \
        --include_base_test --style_test_mode list --style_path ./data/CIFAR10/in_random_style_list_per10 \
        --ckpt_name best_CIFAR10_in_rand_list_per10_styleaug.pth --transform_round $i --content_size 128 --style_size 128 
    done
    for ((i=2; i<=15; i++)); do
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u fusion_test.py --base_test_path ./data/CIFAR10/test \
        --include_base_test --style_test_mode list --style_path ./data/CIFAR10/in_random_style_list_per10 \
        --ckpt_name best_CIFAR10_in_rand_list_per10_styleaug.pth --transform_round $i \
        --content_size 128 --style_size 128 --load_style_test
    done
  ;;
  esac
fi


# === Mixup, Cutout etc. for STL10, uncomment to run ===
# Mixup:
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u extension/train.py --dataset STL10 \
# --train_data_path ./data/STL10-data/train --test_data_path ./data/STL10-data/test \
# --ckpt_name STL10_Mixup.pth --lr 0.001 --milestones 80 120 --gamma 0.1 --max_epoch 150 \
# --train_batch 256 --test_batch 256 --optim Adam --multi_gpu --train_num_workers 4 \
# --tra_augment --no_style_aug --mixup --alpha 1.0

# Cutout:
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u extension/train.py --dataset STL10 \
# --train_data_path ./data/STL10-data/train --test_data_path ./data/STL10-data/test \
# --ckpt_name STL10_Cutout.pth --lr 0.001 --milestones 80 120 --gamma 0.1 --max_epoch 150 \
# --train_batch 256 --test_batch 256 --optim Adam --multi_gpu --train_num_workers 4 \
# --tra_augment --no_style_aug --cutout --n_holes 1 --length 32

# CutMix:
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u extension/train.py --dataset STL10 \
# --train_data_path ./data/STL10-data/train --test_data_path ./data/STL10-data/test \
# --ckpt_name STL10_CutMix.pth --lr 0.001 --milestones 80 120 --gamma 0.1 --max_epoch 150 \
# --train_batch 256 --test_batch 256 --optim Adam --multi_gpu --train_num_workers 4 \
# --tra_augment --no_style_aug --cutmix --beta 1 --cutmix_prob 0.5

# Manifold Mixup:
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u extension/train.py --dataset STL10 \
# --train_data_path ./data/STL10-data/train --test_data_path ./data/STL10-data/test \
# --ckpt_name STL10_CutMix.pth --lr 0.001 --milestones 80 120 --gamma 0.1 --max_epoch 150 \
# --train_batch 256 --test_batch 256 --optim Adam --multi_gpu --train_num_workers 4 \
# --tra_augment --no_style_aug --mixup_hidden --alpha 1 --layer_mix 0 1 2




