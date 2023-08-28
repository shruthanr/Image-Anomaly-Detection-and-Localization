import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import os
from PIL import Image



# This dataset code is inspired (but modified) from: 
# https://github.com/byungjae89/MahalanobisAD-pytorch/blob/master/src/datasets/mvtec.py


class MVTecDataset(Dataset):
    def __init__(self, root_path='.', class_name='carpet', train=True, resize=256, cropsize=224):
        
        self.root_path = root_path
        self.class_name = class_name
        self.train = train
        self.resize = resize
        self.cropsize = cropsize
        self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        
        images, labels, mask = [], [], []

        folder = 'train' if self.train else 'test'
        img_dir = os.path.join(self.mvtec_folder_path, self.class_name, folder)
        gt_dir = os.path.join(self.mvtec_folder_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            
            image_list = []
            for image_file in os.listdir(img_type_dir):
                if not image_file.endswith('.png'):
                    continue
                image_list.append(os.path.join(img_type_dir, image_file))
            image_list.sort()    
            
            images.extend(image_list)

            # load gt labels
            if img_type == 'good':
                labels.extend([0] * len(image_list))
                mask.extend([None] * len(image_list))
            else:
                labels.extend([1] * len(image_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in image_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)



        return list(images), list(labels), list(mask)