import collections
import numbers
import random
import math
from PIL import Image, ImageOps
import torch
import torchvision.transforms.functional as F
import torch.nn.functional as NF

def _iterate_transforms(transforms, x):
    if isinstance(transforms, collections.Iterable):
        for i, transform in enumerate(transforms):
            x[i] = _iterate_transforms(transform, x[i])     
    else :
        if transforms is not None:
            x = transforms(x)   
    return x

# we can pass nested arrays inside Compose
# the first level will be applied to all inputs
# and nested levels are passed to nested transforms
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = _iterate_transforms(transform, x) 
        return x

class RandomHorizontalRollGenerator(object):
    def __call__(self, img):
        self.apply = random.random()
        im,EM,CM =img
        self.roll=random.randint(-im.shape[-1]//2,im.shape[-1]//2)
        return img

class RandomHorizontalFlipGenerator(object):
    def __call__(self, img):
        self.apply = random.random()
        return img        
class RandomGaussianNoiseBlurGenerator:
    def __call__(self, img):
        self.apply = random.random()
        self.size = random.randint(1,10)
        return img
class RandomHorizontalRoll(object):
    def __init__(self, gen, p=0.5):
        self.p = p
        self._gen = gen
    def __call__(self, image):
        if self._gen.apply < self.p:
            return  torch.roll(image,self._gen.roll,dims=-1)
        return image
class RandomHorizontalFlip(object):
    def __init__(self, gen, p=0.5):
        self.p = p
        self._gen = gen
    def __call__(self, image):
        if self._gen.apply < self.p:
            return  F.hflip(image)
        return image   
                     
class RandomGaussianBlur(object) : 
    def __init__(self, gen, p=0.5 ,sigma=2., dim=2, channels=1):
        self.p = p
        self._gen = gen
        self.sigma = sigma
        self.dim = dim
        self.channels = channels
    def gaussian_kernel(self,size):
        # The gaussian kernel is the product of the gaussian function of each dimension.
        # kernel_size should be an odd number.
    
        kernel_size = 2*size + 1

        kernel_size = [kernel_size] * self.dim
        sigma = [self.sigma] * self.dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(self.channels, *[1] * (kernel.dim() - 1))

        return kernel

    def _gaussian_blur(self, x, size=10):
        kernel = self.gaussian_kernel(size=size)
        kernel_size = 2*size + 1

        x = x[None,...]
        padding = int((kernel_size - 1) / 2)
        x = NF.pad(x, (padding, padding, padding, padding), mode='reflect')
        x = torch.squeeze(NF.conv2d(x, kernel, groups=self.channels), dim=0)

        return x

    def __call__(self, image):
        if self._gen.apply < self.p:
            return  self._gaussian_blur(image,size=self._gen.size)
        return image

class RandomGaussianNoise(object) : 
    def __init__(self, gen, p=0.5):
        self.p = p
        self._gen = gen
    def __call__(self,image):
        if self._gen.apply < self.p:
            image = image + torch.randn_like(image)
        return image
