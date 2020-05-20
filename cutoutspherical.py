import torch
import math


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (degrees converted to radians) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = math.radians(length)

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.shape[-2]
        w = img.shape[-1]

        mask = torch.ones(h, w, dtype=torch.float32)

        for n in range(self.n_holes):
            phi = torch.rand((1,1)) * math.pi
            theta = torch.rand((1,1)) * 2 * math.pi

            phi1 = torch.clamp(phi - self.length / 2, 0, math.pi)
            phi2 = torch.clamp(phi + self.length / 2, 0, math.pi)
            theta1 = torch.clamp(theta - self.length / 2, 0, 2*math.pi)
            theta2 = torch.clamp(theta + self.length / 2, 0, 2*math.pi)

            

            x1 = (theta1/(2*math.pi)) * w
            x2 = (theta2/(2*math.pi)) * w
            y1 = (phi1/math.pi) * h
            y2 = (phi2/math.pi) * h

            x1=int(round(x1.item()))
            x2=int(round(x2.item()))
            y1=int(round(y1.item()))
            y2=int(round(y2.item()))
            
            mask[y1: y2, x1: x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask

        return img

if __name__ == '__main__' :
    import torchvision.transforms as transforms
    import PIL.Image as Image
    image = Image.open('img.jpg')
    totensor = transforms.ToTensor()
    image = totensor(image)
    cutout = Cutout(1,60)
    image = cutout(image)
    topil = transforms.ToPILImage()
    image = topil(image)
    image.show()