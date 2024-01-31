
from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random
import os
import abc
import torch
import pickle
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import json

class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'fundus':
            return '../../../../data/disc_cup_split/'  # foler that contains leftImg8bit/
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

class ADAMDataSet(Dataset):
    """
    AMD classification dataset ADAM
    """
    def __init__(self,
                 base_dir=Path.db_root_dir('fundus'),
                 dataset='refuge',
                 split='train',
                 testid=None,
                 transform=None,
                 preprocess = None,
                 num_shot=16
                 ):
        self._base_dir = base_dir
        self.image_list = []
        self.split = split
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.count = 0
        self.amd=0
        self.non=0
        self.preprocess = preprocess

        self._image_dir = os.path.join(self._base_dir, dataset, split)
        print(self._image_dir)
        # imagelist = glob(self._image_dir + "/*.jpg")

        with open(os.path.join(self._base_dir, dataset, "adam_fewshot.json"), "r") as f:
            data_total = json.load(f)
        
        data_train = data_total['train']
        data_test = data_total['test']

        self.shot = num_shot
        if type(self.shot) is int:
            self.shot = str(self.shot)
        
        self.training_set = data_train[self.shot]
        self.test_set = data_test
            
        self.transform = transform
            
        # print('image list ',len(self.image_list))

        print("Number of training images:", len(self.training_set))
        print("Number of testing images:", len(self.test_set))

        if self.split == "Train":
            self.image_list = self.training_set

            for term in self.image_list:
                print(term[0], term[1])
            for term in self.test_set:
                print(term[0], term[1])
                
        elif self.split == "Test":
            self.image_list = self.test_set
        else:
            self.image_list = self.training_set
            # raise NotImplementedError        

        
    def __len__(self):
        return len(self.image_list)

    def get_class_num(self):
        return 2

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "Train" not in self.split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self.split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num
        
        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return  [1.0] * cls_num #weight_list.tolist()


    def __getitem__(self, index):

        _img_name, label = self.image_list[index]
        _img = Image.open(os.path.join(self._image_dir, _img_name)).convert("RGB")
                    
        if self.transform is not None:
            _img = self.transform(_img)
        
        if self.split == 'Train':
            report_pth = self._image_dir.replace('/Train','/english.txt')
            txt = 'a photo of normal eye'
            with open(report_pth, 'r', encoding='utf8') as file:
                reader = file.readlines() # csv.reader(file)
                for line in reader:
                    if line.split()[0] == _img_name.split('.')[0]:
                        separator = ' '
                        txt = separator.join(line.split()[1:])
                        txt = 'a photo of age-related macular degeneration eye with drusen or disorder ' + txt
            anco_sample = {'image': _img, 'label': label, 'img_name': _img_name, 'txt': txt}
        elif self.split == 'Test':
            if label == 0:
                txt = 'a photo of normal eye'
            else:
                txt = 'a photo of eye with age-related macular degeneration disease with drusen or disorder'
            anco_sample = {'image': _img, 'label': label, 'img_name': _img_name, 'txt': txt}
        else:
            anco_sample = {'image': _img, 'label': label, 'img_name': _img_name}

        return anco_sample

    def __str__(self):
        return 'ADAM(split=' + str(self.split) + ')'


class ODIRDataSet(Dataset):
    """
    AMD classification dataset ODIR-5k
    """
    def __init__(self,
                 base_dir=Path.db_root_dir('fundus'),
                 dataset='refuge',
                 split='train',
                 testid=None,
                 transform=None,
                 num_shot=16,
                 ):
        self._base_dir = base_dir
        self.image_list = []
        self.split = split
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []  

        self._image_dir = os.path.join(self._base_dir, dataset, split)
        print(self._image_dir)
        # imagelist = glob(self._image_dir + "/*.jpg")

        with open(os.path.join(self._base_dir, dataset, "odir_fewshot.json"), "r") as f:
            data_total = json.load(f)
        
        data_train = data_total['train']
        data_test = data_total['test']

        self.shot = num_shot
        if type(self.shot) is int:
            self.shot = str(self.shot)
        
        self.training_set = data_train[self.shot]
        self.test_set = data_test
            
        self.transform = transform
            

        print("Number of training images:", len(self.training_set))
        print("Number of testing images:", len(self.test_set))

        if self.split == "Train":
            self.image_list = self.training_set

            for term in self.image_list:
                print(term[0], term[1])
            for term in self.test_set:
                print(term[0], term[1])
                
        elif self.split == "Test":
            self.image_list = self.test_set
        else:
            self.image_list = self.training_set
            # raise NotImplementedError   

    def __len__(self):
        return len(self.image_list)
    
    def get_class_num(self):
        return 2

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "Train" not in self.split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self.split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num
        
        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return  [1.0] * cls_num #weight_list.tolist()

    def __getitem__(self, index):
        _img_name, label = self.image_list[index]
        _img = Image.open(os.path.join(self._image_dir, _img_name)).convert("RGB")
                    
        if self.transform is not None:
            _img = self.transform(_img)
            
        if self.split == 'Train':
            report_pth = self._image_dir.replace('/Train','/english.txt')
            txt = 'a photo of normal eye'
            with open(report_pth, 'r', encoding='utf8') as file:
                reader = file.readlines() #csv.reader(file)
                for line in reader:
                    if line.split()[0] == _img_name.split('.')[0]:
                        separator = ' '
                        txt = separator.join(line.split()[1:])
                        txt = 'a photo of age-related macular degeneration eye ' + txt
            anco_sample = {'image': _img, 'label': label, 'img_name': _img_name, 'txt': txt}
        else:
            anco_sample = {'image': _img, 'label': label, 'img_name': _img_name}
        
        
        return anco_sample

    def __str__(self):
        return 'RIADD(split=' + str(self.split) + ')'

class ARIADataSet(Dataset):
    """
    AMD classification dataset ARIA
    """
    def __init__(self,
                 base_dir=Path.db_root_dir('fundus'),
                 dataset='refuge',
                 split='train',
                 testid=None,
                 transform=None,
                 num_shot=16,
                 ):
        self._base_dir = base_dir
        self.image_list = []
        self.split = split
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.amd = 0
        self.none = 0
        self.count = 0

        self._image_dir = os.path.join(self._base_dir, dataset, split)
        print(self._image_dir)

        # imagelist = glob(self._image_dir + "/*.jpg")

        with open(os.path.join(self._base_dir, dataset, "aria_fewshot.json"), "r") as f:
            data_total = json.load(f)
        
        data_train = data_total['train']
        data_test = data_total['test']

        self.shot = num_shot
        if type(self.shot) is int:
            self.shot = str(self.shot)
        
        self.training_set = data_train[self.shot]
        self.test_set = data_test
            
        self.transform = transform
            
        # print('image list ',len(self.image_list))

        print("Number of training images:", len(self.training_set))
        print("Number of testing images:", len(self.test_set))

        if self.split == "Train":
            self.image_list = self.training_set
            for term in self.image_list:
                print(term[0], term[1])
            for term in self.test_set:
                print(term[0], term[1])
        elif self.split == "Test":
            self.image_list = self.test_set
        else:
            self.image_list = self.training_set
            # raise NotImplementedError        



    def __len__(self):
        return len(self.image_list)

    def get_class_num(self):
        return 2

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "Train" not in self.split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self.split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num
        
        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return  [1.0] * cls_num #weight_list.tolist()


    def __getitem__(self, index):

        _img_name, label = self.image_list[index]
        _img = Image.open(os.path.join(self._image_dir, _img_name)).convert("RGB")
                    
        if self.transform is not None:
            _img = self.transform(_img)

        if self.split == 'Train':
            report_pth = self._image_dir.replace('/Train','/english.txt')
            txt = 'a photo of normal eye'
            with open(report_pth, 'r', encoding='utf8') as file:
                reader = file.readlines() #csv.reader(file)
                for line in reader:
                    if line.split()[0] == _img_name.split('.')[0]:
                        separator = ' '
                        txt = separator.join(line.split()[1:])
                        txt = 'a photo of age-related macular degeneration eye ' + txt
            anco_sample = {'image': _img, 'label': label, 'img_name': _img_name, 'txt': txt}
        else:
            anco_sample = {'image': _img, 'label': label, 'img_name': _img_name}
        
        return anco_sample

    def __str__(self):
        return 'ARIA(split=' + str(self.split) + ')'

class STAREDataSet(Dataset):
    """
    AMD classification dataset STARE
    """
    def __init__(self,
                 base_dir=Path.db_root_dir('fundus'),
                 dataset='refuge',
                 split='train',
                 testid=None,
                 transform=None,
                 num_shot=16
                 ):
        self._base_dir = base_dir
        self.image_list = []
        self.split = split
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []  
        self._image_dir = os.path.join(self._base_dir, dataset, split)
        self.count = 0

        self._image_dir = os.path.join(self._base_dir, dataset, split)
        print(self._image_dir)

        # imagelist = glob(self._image_dir + "/*.jpg")

        with open(os.path.join(self._base_dir, dataset, "stare_fewshot.json"), "r") as f:
            data_total = json.load(f)
        
        data_train = data_total['train']
        data_test = data_total['test']

        self.shot = num_shot
        if type(self.shot) is int:
            self.shot = str(self.shot)
        
        self.training_set = data_train[self.shot]
        self.test_set = data_test
            
        self.transform = transform
            
        # print('image list ',len(self.image_list))

        print("Number of training images:", len(self.training_set))
        print("Number of testing images:", len(self.test_set))

        if self.split == "Train":
            self.image_list = self.training_set
        elif self.split == "Test":
            self.image_list = self.test_set
        else:
            self.image_list = self.training_set
            # raise NotImplementedError    

    def __len__(self):
        return len(self.image_list)

    def get_class_num(self):
        return 2

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "Train" not in self.split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self.split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num
        
        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return  [1.0] * cls_num #weight_list.tolist()


    def __getitem__(self, index):
        _img_name, label = self.image_list[index]
        _img = Image.open(os.path.join(self._image_dir, _img_name)).convert("RGB")
                    
        if self.transform is not None:
            _img = self.transform(_img)

        if self.split == 'Train':
            report_pth = self._image_dir.replace('/Train','/english.txt')
            txt = 'a photo of normal eye'
            with open(report_pth, 'r', encoding='utf8') as file:
                reader = file.readlines() #csv.reader(file)
                for line in reader:
                    if line.split()[0] == _img_name.split('.')[0]:
                        separator = ' '
                        txt = separator.join(line.split()[1:])
                        txt = 'a photo of age-related macular degeneration eye ' + txt
            anco_sample = {'image': _img, 'label': label, 'img_name': _img_name, 'txt': txt}
        else:
            anco_sample = {'image': _img, 'label': label, 'img_name': _img_name}
        

        return anco_sample

    def __str__(self):
        return 'STARE(split=' + str(self.split) + ')'


if __name__ == '__main__':
    data_dir = '/data/yedu/FSL/AMD_Classification/AMD_Classification/'
    dataset = 'ADAM' #'ARIA' #'ADAM'ODIR
    composed_transforms_train = transforms.Compose([
            tr.Resize(512),
            #tr.RandomFlip(),
            # tr.add_salt_pepper_noise(),
            # tr.adjust_light(),
            tr.eraser(),
            tr.Normalize_tf(),
            tr.ToTensor()
        ])
    composed_transforms_test = transforms.Compose([
        tr.Resize(512),
        #tr.RandomFlip(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    # # RIADD
    # db_train = RiaddDataSet(base_dir=data_dir, dataset=dataset, split='Training_Set/train', transform=composed_transforms_train)
    # db_test = RiaddDataSet(base_dir=data_dir, dataset=dataset, split='Test_Set/test', transform=composed_transforms_test)

    # ADAM
    db_train = ADAMDataSet(base_dir=data_dir, dataset=dataset, split='Train', transform=composed_transforms_train)
    db_test = ADAMDataSet(base_dir=data_dir, dataset=dataset, split='Test', transform=composed_transforms_test)
    # db_v = ADAMDataSet(base_dir=data_dir, dataset=dataset, split='Validation', transform=composed_transforms_test)

    # ODIR-5k
    # db_train = ODIRDataSet(base_dir=data_dir, dataset=dataset, split='Train', transform=composed_transforms_train)
    # db_test = ODIRDataSet(base_dir=data_dir, dataset=dataset, split='Test', transform=composed_transforms_test)

    # ARIA
    # db_train = ARIADataSet(base_dir=data_dir, dataset=dataset, split='Train', transform=composed_transforms_train)
    # db_test = ARIADataSet(base_dir=data_dir, dataset=dataset, split='Test', transform=composed_transforms_test)

    # STARE
    # db_train = STAREDataSet(base_dir=data_dir, dataset=dataset, split='Train', transform=composed_transforms_train)
    # db_test = STAREDataSet(base_dir=data_dir, dataset=dataset, split='Test', transform=composed_transforms_test)

    train_loader = DataLoader(db_train, batch_size=8, shuffle=True, num_workers=1)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    for batch_idx, (sample) in enumerate(train_loader):
            data, label, img_name = sample['image'], sample['label'], sample['img_name']
            print(label)
    for batch_idx, (sample) in enumerate(test_loader):
            data, label, img_name = sample['image'], sample['label'], sample['img_name']
            # print(label)

    
    # plt.figure(figsize=(12,7))
    # for i in range(10):
    #     sample = random.choice(range(len(db_train)))
    #     image = db_train[sample]['image']
    #     label = db_train[sample]['label']
    #     image = np.asarray(image)
    #     print(image[:,125:126,125:126], label)
    #     if label== 0:
    #         label = "Non-AMD"
    #     else:
    #         label = "AMD"
    #     plt.subplot(2,5,i+1)
    #     plt.imshow(255*image.T)
    #     plt.xlabel(label)
    # plt.tight_layout()    