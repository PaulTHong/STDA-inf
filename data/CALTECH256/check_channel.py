"""
Pick image whose channel is 1, not 3. 
torchvision.transforms.ToTensor() can handle with this phenomenon.
"""
import os
import cv2

pre_path = './train'

classes = os.listdir(pre_path)
store = []
for cls in classes:
    path = os.path.join(pre_path, cls)
    print('Start ', cls)
    imgs = os.listdir(path)
    for img_p in imgs:
        img = cv2.imread(os.path.join(path, img_p), cv2.IMREAD_UNCHANGED)
        print(img_p, img.shape)
        if len(img.shape)!=3 or img.shape[2]!=3:
            store.append((os.path.join(cls, img_p), img.shape))

print('See strange:')
if len(store) != 0:
    for s in store:
        print(s)

with open('strange.txt', 'w') as f:
    if len(store) != 0:
        for s in store:
            f.write(s[0] + ' ' + str(s[1]) + '\n')
