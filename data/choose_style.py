'''
Choose in-data style images from the train dataset.
list (class-agnostic) or dict (in-class) 
'''
import os
import numpy as np

DATASET = 'STL10'

if DATASET == 'STL10':
    source_path = './STL10-data/train' 
    num_per_class = 10
    classes = tuple([str(i) for i in range(1, 11)])
    list_target = './STL10-data/stl_random_style_list_per10'
    dict_target = './STL10-data/stl_random_style_dict_per10'
else:
    raise NotImplementedError()

def choose_style_list(target):
    if not os.path.exists(target):
        os.mkdir(target)
    for cls in classes:
        source = os.path.join(source_path, cls)
        img_list = os.listdir(source)   
        imgs = np.random.choice(img_list, num_per_class, replace=False)    
        for img in imgs:
            print(img)
            os.system('cp %s %s' %(os.path.join(source, img), target))


def choose_style_dict(target):
    if not os.path.exists(target):
        os.mkdir(target)
    for cls in classes:
        cls_path = os.path.join(target, cls)
        if not os.path.exists(cls_path):
            os.system('mkdir -p %s' % (cls_path))
        
        source = os.path.join(source_path, cls)
        img_list = os.listdir(source)   
        imgs = np.random.choice(img_list, num_per_class, replace=False)    
        for img in imgs:
            print(img)
            os.system('cp %s %s' %(os.path.join(source, img), cls_path))
    
if __name__ == '__main__':
    choose_style_list(list_target)
    print('==========')
    choose_style_dict(dict_target)


