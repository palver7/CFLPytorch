import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import itertools

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

img_size = [128,256]

train_transform = transforms.Compose(
        [transforms.Resize((img_size[0],img_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
train_target_transform = transforms.Compose([transforms.Resize((img_size[0],img_size[1])),
                                        transforms.ToTensor()])

trainset=SUN360Dataset('traindata.json',transform=train_transform,target_transform=train_target_transform,joint_transform=None) 
supplement= SUN360Dataset('morethan4corners.json',transform=train_transform,target_transform=train_target_transform,joint_transform=None)

train_loader = DataLoader(trainset, batch_size=4-1,
                                               shuffle=True, num_workers=2)
suppl_loader = DataLoader(supplement, batch_size=1,
                                               shuffle=True, num_workers=2)
                                             
for i, data in enumerate(train_loader):
    RGB,EM,CM = data
    RGBsup,EMsup,CMsup = next(itertools.cycle(suppl_loader))
    RGB = torch.cat([RGB,RGBsup],dim=0)
    EM = torch.cat([EM,EMsup],dim=0)
    CM = torch.cat([CM,CMsup],dim=0)
    image = CM[3]
    tojpg = transforms.ToPILImage()
    image = torch.squeeze(image)
    image = tojpg(image)
    image.show()
    break
    