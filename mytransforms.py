import collections
import numbers
import random
import math
from PIL import Image, ImageOps
import torch
import torchvision.transforms.functional as F
import torch.nn.functional as NF
import panostretch
import numpy as np

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

def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius=25, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  #x, y = int(center[0]), int(center[1])
  for x,y in center:  
    x, y = int(np.rint(x)), int(np.rint(y))
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap

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
        im,_,_,_ =img
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
        
class RandomPanoStretchGenerator(object):
    def __init__(self,max_stretch=2.0):
        self.max_stretch = max_stretch

    def cor2xybound(self, cor):
        ''' Helper function to clip max/min stretch factor '''
        corU = cor[0::2]
        corB = cor[1::2]
        zU = -50
        u = panostretch.coorx2u(corU[:, 0])
        vU = panostretch.coory2v(corU[:, 1])
        vB = panostretch.coory2v(corB[:, 1])

        x, y = panostretch.uv2xy(u, vU, z=zU)
        c = np.sqrt(x**2 + y**2)
        zB = c * np.tan(vB)
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        S = 3 / abs(zB.mean() - zU)
        dx = [abs(xmin * S), abs(xmax * S)]
        dy = [abs(ymin * S), abs(ymax * S)]

        return min(dx), min(dy), max(dx), max(dy)

    def __call__(self, img):
        self.apply = random.random()
        _,_,_,cor =img
        self.cor = cor
        xmin, ymin, xmax, ymax = self.cor2xybound(self.cor)
        kx = np.random.uniform(1.0, self.max_stretch)
        ky = np.random.uniform(1.0, self.max_stretch)
        if np.random.randint(2) == 0:
            kx = max(1 / kx, min(0.5 / xmin, 1.0))
        else:
            kx = min(kx, max(10.0 / xmax, 1.0))
        if np.random.randint(2) == 0:
            ky = max(1 / ky, min(0.5 / ymin, 1.0))
        else:
            ky = min(ky, max(10.0 / ymax, 1.0))
        self.kx = kx
        self.ky = ky    
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
    def __init__(self, gen, p=0.5,alpha=0.1):
        self.p = p
        self.alpha = alpha
        self._gen = gen
    def __call__(self,image):
        if self._gen.apply < self.p:
            image = image + self.alpha * torch.randn_like(image)
        return image

class RandomPanoStretch(object):
    def __init__(self, gen, p=0.5):
        self.p = p
        self._gen = gen
    
    def __call__(self, image):
        if self._gen.apply < self.p:
            image = np.asarray(image)
            if image.ndim < 3:
                image = np.expand_dims(image,axis=-1)    
            image, cor = panostretch.pano_stretch(image, self._gen.cor, self._gen.kx, self._gen.ky)
            image = image.astype(np.uint8)
            if image.shape[-1] == 1 :
                image = np.squeeze(image)
            image = Image.fromarray(image)
            if image.mode !='RGB':
                image=image.convert('L')
        
        return image

class RandomPanoStretchCorners(object):
    def __init__(self, gen, p=0.5):
        self.p = p
        self._gen = gen
    
    def __call__(self, image):
        if self._gen.apply < self.p:
            image = np.asarray(image)
            if image.ndim < 3:
                image = np.expand_dims(image,axis=-1)    
            image, cor = panostretch.pano_stretch(image, self._gen.cor, self._gen.kx, self._gen.ky)
            hm = np.zeros((512, 1024), dtype=np.float32)
            hm = (draw_umich_gaussian(hm,cor)*255).astype(np.uint8)
            image = Image.fromarray(hm)    

        return image              

class ImagePreprocessing(object):
    def __call__(self, image):
        image = np.array(image).astype(np.float32)
        if image.ndim < 3:
            image = np.expand_dims(image,axis=-1)    
        if image.shape[-1] == 3:
            mean_color = [103.939, 116.779, 123.68]
            r,g,b = np.split(image,3,axis=2)
            image = np.concatenate((b - mean_color[0], g - mean_color[1], r - mean_color[2]),axis=2)
        tensor = torch.from_numpy(image)
        tensor = tensor.permute(2,0,1)        

        return tensor 