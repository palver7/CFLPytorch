"""This script is for experimenting with custom data transforms with classes on custom datasets"""

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import mytransforms
import os


class SUN360Dataset(Dataset):
    

    def __init__(self, file, transform=None, target_transform=None, joint_transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            transform (callable, optional): Optional transform to be applied
                on an image.
            target_file (callable, optional): Optional transform to be applied
                on a map (edge and corner).    
        """
    
        self.images_data = pd.read_json(file)    
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images_data.iloc[idx, 0]                        
        EM_name = self.images_data.iloc[idx, 1]
        CM_name = self.images_data.iloc[idx, 2]
        CL_name = self.images_data.iloc[idx, 3]
        image = Image.open(img_name)
        EM = Image.open(EM_name)
        CM = Image.open(CM_name)
        with open(CL_name, mode='r') as f:
            cor = np.array([line.strip().split() for line in f], np.int32)
        if(len(cor)%2 != 0) :
            print (CL_name.split('/')[-1])    
        
        """
        EM = np.asarray(EM)
        EM = np.expand_dims(EM, axis=2)
        CM = np.asarray(CM) 
        CM = np.expand_dims(CM, axis=2) 
        gt = np.concatenate((EM,CM),axis = 2)
        maps = Image.fromarray(gt)
        """
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            CM = self.target_transform(CM)
            EM = self.target_transform(EM)
        if self.joint_transform is not None:   
            image, EM, CM, cor = self.joint_transform([image, EM, CM, cor])      
        
        return image, EM, CM

class SplitDataset(Dataset):
    

    def __init__(self, dataset, transform=None, target_transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            transform (callable, optional): Optional transform to be applied
                on an image.
            target_file (callable, optional): Optional transform to be applied
                on a map (edge and corner).    
        """
    
        self.images_data = dataset 
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        
        image, EM, CM = self.images_data[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            CM = self.target_transform(CM)
            EM = self.target_transform(EM)    
        
        return image, EM, CM


#transform = transforms.Compose([transforms.ToTensor(),HorizontalRotation()])
#target_transform = transforms.Compose([transforms.ToTensor()])
roll_gen = mytransforms.RandomHorizontalRollGenerator()
flip_gen = mytransforms.RandomHorizontalFlipGenerator()
panostretch_gen = mytransforms.RandomPanoStretchGenerator(max_stretch = 2.0)
joint_transform = mytransforms.Compose([panostretch_gen,
                                       [mytransforms.RandomPanoStretch(panostretch_gen), mytransforms.RandomPanoStretch(panostretch_gen), mytransforms.RandomPanoStretch(panostretch_gen), None],
                                       flip_gen,
                                       [mytransforms.RandomHorizontalFlip(flip_gen),mytransforms.RandomHorizontalFlip(flip_gen),mytransforms.RandomHorizontalFlip(flip_gen), None],
                                       [transforms.ToTensor(),transforms.ToTensor(),transforms.ToTensor(), None],
                                       roll_gen,
                                       [mytransforms.RandomHorizontalRoll(roll_gen),mytransforms.RandomHorizontalRoll(roll_gen),mytransforms.RandomHorizontalRoll(roll_gen), None],
                                       [transforms.RandomErasing(p=0.5,value=0), None, None, None],
                                       ])

trainset = SUN360Dataset(file="traindata.json",transform = None, target_transform = None, joint_transform=joint_transform)
train_loader = DataLoader(trainset, batch_size=1,
                                               shuffle=True, num_workers=2)    
topil=transforms.ToPILImage()
                                              
if not os.path.exists('result/RGB/'):
    os.makedirs('result/RGB/')
if not os.path.exists('result/EM/'):
    os.makedirs('result/EM/')
if not os.path.exists('result/CM/'):
    os.makedirs('result/CM/')  
        
for i, data in enumerate(train_loader):
    images, EM, CM, cor = data 
    images, EM, CM = torch.squeeze(images), torch.squeeze(EM), torch.squeeze(CM)
    im,edges,corners = topil(images), topil(EM), topil(CM)
    if len (str(i))<2:
        num = '0'+str(i)
    else:
        num = str(i)    
    im.save("result/RGB/RGB_{}.jpg".format(num))
    edges.save("result/EM/EM_{}.jpg".format(num))
    corners.save("result/CM/CM_{}.jpg".format(num))
    
