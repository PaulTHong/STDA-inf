import numpy as np
import cv2
import os

mean = []
std = []
img_list = []

dir_path = './STL10-data/train'
class_paths = os.listdir(dir_path)
print(class_paths)
for cls in class_paths:
    img_paths = os.listdir(dir_path + os.sep + cls)
    print(len(img_paths))
    for img_path in img_paths:
        print(img_path)
        img_path = dir_path + os.sep + cls + os.sep + img_path
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = img[::, np.newaxis]
        img_list.append(img)

# dir_path = './STL10-data/test'
# class_paths = os.listdir(dir_path)
# print(class_paths)
# for cls in class_paths:
    # img_paths = os.listdir(dir_path + os.sep + cls)
    # print(len(img_paths))
    # for img_path in img_paths:
        # print(img_path)
        # img_path = dir_path + os.sep + cls + os.sep + img_path
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # img = img[::, np.newaxis]
        # img_list.append(img)

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.0

for i in range(3):
    channel = imgs[:, :, i, :].ravel()
    mean.append(np.mean(channel))
    std.append(np.std(channel))

mean.reverse()
std.reverse()

print(mean)
print(std)





