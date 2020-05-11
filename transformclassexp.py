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
        image = Image.open(img_name)
        EM = Image.open(EM_name)
        CM = Image.open(CM_name)
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
            image, EM, CM = self.joint_transform([image, EM, CM])      
        
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

class RandomHorizontalRoll(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return  torch.roll(image,random.randint(-image.shape[-1]//2,image.shape[-1]//2),dims=-1)
        return image    
    

#transform = transforms.Compose([transforms.ToTensor(),HorizontalRotation()])
#target_transform = transforms.Compose([transforms.ToTensor()])
roll_gen = mytransforms.RandomHorizontalRollGenerator()
flip_gen = mytransforms.RandomHorizontalFlipGenerator()
noiseblur_gen = mytransforms.RandomGaussianNoiseBlurGenerator()
joint_transform = mytransforms.Compose([[transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1), None, None],
                                       flip_gen,
                                       [mytransforms.RandomHorizontalFlip(flip_gen),mytransforms.RandomHorizontalFlip(flip_gen),mytransforms.RandomHorizontalFlip(flip_gen)],
                                       [transforms.ToTensor(),transforms.ToTensor(),transforms.ToTensor()],
                                       roll_gen,
                                       [mytransforms.RandomHorizontalRoll(roll_gen),mytransforms.RandomHorizontalRoll(roll_gen),mytransforms.RandomHorizontalRoll(roll_gen)],
                                       noiseblur_gen,
                                       [mytransforms.RandomGaussianNoise(noiseblur_gen),mytransforms.RandomGaussianBlur(noiseblur_gen), mytransforms.RandomGaussianBlur(noiseblur_gen)],
                                       [transforms.RandomErasing(p=0.8,scale=(0.01,0.02),ratio=(0.3,3.3),value='random'), None, None],
                                       [transforms.RandomErasing(p=0.8,scale=(0.02,0.03),ratio=(0.3,3.3),value='random'), None, None],
                                       [transforms.RandomErasing(p=0.6,scale=(0.03,0.04),ratio=(0.3,3.3),value='random'), None, None],
                                       [transforms.RandomErasing(p=0.6,scale=(0.04,0.05),ratio=(0.3,3.3),value='random'), None, None],
                                       [transforms.RandomErasing(p=0.4,scale=(0.05,0.06),ratio=(0.3,3.3),value='random'), None, None],
                                       [transforms.RandomErasing(p=0.4,scale=(0.06,0.07),ratio=(0.3,3.3),value='random'), None, None], 
                                       [transforms.RandomErasing(p=0.2,scale=(0.07,0.08),ratio=(0.3,3.3),value='random'), None, None],
                                       [transforms.RandomErasing(p=0.2,scale=(0.08,0.09),ratio=(0.3,3.3),value='random'), None, None],
                                       [transforms.RandomErasing(p=0.1,scale=(0.09,0.10),ratio=(0.3,3.3),value='random'), None, None],
                                       [transforms.RandomErasing(p=0.1,scale=(0.1,0.11),ratio=(0.3,3.3),value='random'),  None, None]])

trainset = SUN360Dataset(file="traindatasmall.json",transform = None, target_transform = None, joint_transform=joint_transform)
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
    images, EM, CM = data 
    images, EM, CM = torch.squeeze(images), torch.squeeze(EM), torch.squeeze(CM)
    im,edges,corners = topil(images), topil(EM), topil(CM)
    if len (str(i))<2:
        num = '0'+str(i)
    else:
        num = str(i)    
    im.save("result/RGB/RGB_{}.jpg".format(num))
    edges.save("result/EM/EM_{}.jpg".format(num))
    corners.save("result/CM/CM_{}.jpg".format(num))
    
