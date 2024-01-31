
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
                 num_shot=-1
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
        imagelist = glob(self._image_dir + "/*.jpg")

        if self.split == 'Train':
            print("total train list:", len(imagelist))
            print("total train imagelist:", imagelist)

        if num_shot > 0:
            trains = imagelist

            trains_cls_1 = [i for i in trains if "A" in i.split("/")[-1]]
            trains_cls_2 = [i for i in trains if "N" in i.split("/")[-1]]

            trains_cls_1 = trains_cls_1[:num_shot]
            trains_cls_2 = trains_cls_2[:num_shot]

            print("*" * 100)
            print("list_of_trains_cls_1:", len(trains_cls_1))
            print("list_of_trains_cls_2:", len(trains_cls_2))
            print("list_of_Trains_cls_1:", trains_cls_1)
            print("list_of_Trains_cls_1:", trains_cls_2)
            print("*" * 100)



        if self.split == 'Train':
            for image_path in imagelist:
                if image_path in trains_cls_1 or image_path in trains_cls_2:
                    self.image_list.append({'image': image_path, 'id': testid})
            
            print("*" * 100)
            print('image training list ',len(self.image_list))
            print("self.image_list:", self.image_list)
            print("*" * 100)
                
        if self.split == 'Test':
            for image_path in imagelist: 
                self.image_list.append({'image': image_path, 'id': testid})
            
        self.transform = transform
            
        print('image list ',len(self.image_list))
        
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
        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _img_name = self.image_list[index]['image'].split('/')[-1]

        if 'A' in _img_name: # amd
            label = 1
            self.amd += 1  

        elif 'N' in _img_name: # non-amd
            label = 0 
            self.non +=1
        self.count += 1

        if self.split == 'Test':
            txt_pth = self._image_dir.replace('/Test','/Test/test_classification_GT.txt')#('/Validation','/Validation/validation_classification_GT.txt')
            label_file = open(txt_pth, 'r')
            # only consider AMD and non-AMD
            img_total = {}
            data = label_file.readlines()
            for line in data:
                if line.split()[0] == _img_name:
                    label = int(line.split()[1])
                    
        if self.split == 'Validation':
            txt_pth = self._image_dir.replace('/Validation','/Validation/validation_classification_GT.txt')
            label_file = open(txt_pth, 'r')
            # only consider AMD and non-AMD
            img_total = {}
            data = label_file.readlines()
            for line in data:
                if line.split()[0] == _img_name:
                    label = int(line.split()[1])
                    
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
                 transform=None

                 ):
        self._base_dir = base_dir
        self.image_list = []
        self.split = split
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []  
        self._image_dir = os.path.join(self._base_dir, dataset, split)
        self.count = 0

        imagelist = glob(self._image_dir + "/*.jpg")
        for image_path in imagelist:
            self.image_list.append({'image': image_path, 'id': testid})
        
        self.transform = transform

        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

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
        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _img_name = self.image_list[index]['image'].split('/')[-1]

        if self.split == 'Train':
            csv_pth = self._image_dir.replace('/Train','/full_df.csv')
        elif self.split == 'Test':
            csv_pth = self._image_dir.replace('/Test','/full_df.csv')

        data_ = pd.read_csv(csv_pth)

        # only consider AMD and non-AMD
        number_str = _img_name.split('_')[0]
        index_t = int(number_str)
        if 'left' in _img_name:
            left_amd = data_.loc[index_t+1, "Left-Diagnostic Keywords"]
            if 'age' in left_amd:
                label = 1
            else:
                label = 0
            
        elif 'right' in _img_name:
            right_amd = data_.loc[index_t+1, "Right-Diagnostic Keywords"]
            if 'age' in right_amd:
                label = 1
            else:
                label = 0

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
                 transform=None

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
        imagelist = glob(self._image_dir + "/*.tif")
        for image_path in imagelist:
            self.image_list.append({'image': image_path, 'id': testid})

        self.transform = transform
      
        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

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
        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        _img_name = self.image_list[index]['image'].split('/')[-1]
        _class = _img_name.split('_')[1]
        
        if _class == 'a': # amd
            label = 1
            # self.amd += 1
        else:
            label = 0 # non-amd
            # self.none += 1
        if 'aug' in self.image_list[index]['image']:
            label = 1
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
                 transform=None

                 ):
        self._base_dir = base_dir
        self.image_list = []
        self.split = split
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []  
        self._image_dir = os.path.join(self._base_dir, dataset, split)
        self.count = 0

        imagelist = glob(self._image_dir + "/*.ppm")
            
        for image_path in imagelist:
            self.image_list.append({'image': image_path, 'id': testid})
        
        self.transform = transform
        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

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
        _img = Image.open(self.image_list[index]['image']).convert('RGB')
        
        _img_name = self.image_list[index]['image'].split('/')[-1]

        if self.split == 'Train':
            txt_pth = self._image_dir.replace('/Train','/all-mg-codes.txt')
        elif self.split == 'Test':
            txt_pth = self._image_dir.replace('/Test','/all-mg-codes.txt')
        label_file = open(txt_pth, 'r')

        # only consider AMD and non-AMD
        img_total = {}
        data = label_file.readlines()
        for line in data:
            im_id = line.split()[0]
            _disease = line.split()[1:]
            if 'Age' in _disease:
                _label = 1
            else:
                _label = 0
                # self.non += 1
            img_total.update({im_id: _label})

        for key in img_total.keys():
            if _img_name.split('.')[0] == key:
                label = img_total[key]
                if label == 1:
                    self.count += label
                
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
    data_dir = '/mnt/data1/llr_data/AMD_Classification/'
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
    # db_train = ADAMDataSet(base_dir=data_dir, dataset=dataset, split='Train', transform=composed_transforms_train)
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

    # train_loader = DataLoader(db_train, batch_size=8, shuffle=True, num_workers=1)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    # for batch_idx, (sample) in enumerate(train_loader):
    #         data, label, img_name = sample['image'], sample['label'], sample['img_name']
    #         print(label)
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