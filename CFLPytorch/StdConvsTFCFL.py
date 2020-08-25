import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from torchvision.models.utils import load_state_dict_from_url

class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        #self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])    
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, return_indices = False, ceil_mode = False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        #self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size, self.kernel_size 
        sh, sw = self.stride, self.stride
        dh, dw = self.dilation, self.dilation
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])    
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)

class ConvTranspose2dDynamicSamePadding(nn.ConvTranspose2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        #self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        oh, ow = math.ceil(ih * sh), math.ceil(iw * sw)
        pad_h = max(((ih - 1) * sh + (kh - 1) * dh + 1 - oh), 0)
        pad_w = max(((iw - 1) * sw + (kw - 1) * dw + 1 - ow), 0)
        x = F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        if pad_h > 0 or pad_w > 0:
            xh, xw = x.size()[-2:]
            x = x[:,:,1: xh - (pad_h - pad_h // 2), pad_w // 2: xw - (pad_w - pad_w // 2)]    
            
        return x  

Conv2d = Conv2dDynamicSamePadding
MaxPool2d = MaxPool2dDynamicSamePadding
ConvTranspose2d = ConvTranspose2dDynamicSamePadding

class StdConvsTFCFL(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.relu = nn.ReLU()
        bn_mom = 1 - 0.999
        bn_eps = 1e-3

        #self.feed('rgb_input')
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=True)
        self.bn_conv1 = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_mom)
        self.pool1 = MaxPool2d(3, stride=2, padding=0)
        self.res2a_branch1 = Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn2a_branch1 = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        
        #self.feed('pool1')
        self.res2a_branch2a = Conv2d(64, 64, kernel_size=1, stride=1, bias=False)
        self.bn2a_branch2a = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_mom)
        self.res2a_branch2b = Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2a_branch2b = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_mom)
        self.res2a_branch2c = Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn2a_branch2c = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        
        #self.feed('bn2a_branch1', 
        #           'bn2a_branch2c')
        #         .add(name='res2a')
        self.res2a_relu = nn.ReLU()
        self.res2b_branch2a = Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.bn2b_branch2a = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_mom)
        self.res2b_branch2b = Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2b_branch2b = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_mom)
        self.res2b_branch2c = Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn2b_branch2c = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        
        #self.feed('res2a_relu', 
        #           'bn2b_branch2c')
        #         .add(name='res2b')
        self.res2b_relu = nn.ReLU()
        self.res2c_branch2a = Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.bn2c_branch2a = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_mom)
        self.res2c_branch2b = Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2c_branch2b = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_mom)
        self.res2c_branch2c = Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn2c_branch2c = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        
        #self.feed('res2b_relu', 
        #           'bn2c_branch2c')
        #         .add(name='res2c')
        self.res2c_relu=nn.ReLU()
        self.res3a_branch1 = Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.bn3a_branch1 = nn.BatchNorm2d(512, eps=bn_eps, momentum=bn_mom)

        #(self.feed('res2c_relu')
        self.res3a_branch2a = Conv2d(256, 128, kernel_size=1, stride=2, bias=False)
        self.bn3a_branch2a = nn.BatchNorm2d(128, eps=bn_eps, momentum=bn_mom)
        self.res3a_branch2b = Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3a_branch2b = nn.BatchNorm2d(128, eps=bn_eps, momentum=bn_mom)
        self.res3a_branch2c = Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn3a_branch2c =  nn.BatchNorm2d(512, eps=bn_eps, momentum=bn_mom)

        #self.feed('bn3a_branch1', 
        #           'bn3a_branch2c')
        #         .add(name='res3a')
        self.res3a_relu = nn.ReLU()
        self.res3b_branch2a = Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.bn3b_branch2a = nn.BatchNorm2d(128, eps=bn_eps, momentum=bn_mom)
        self.res3b_branch2b = Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3b_branch2b = nn.BatchNorm2d(128, eps=bn_eps, momentum=bn_mom)
        self.res3b_branch2c = Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn3b_branch2c = nn.BatchNorm2d(512, eps=bn_eps, momentum=bn_mom)

        #self.feed('res3a_relu', 
        #        'bn3b_branch2c')
        #        .add(name='res3b')
        self.res3b_relu = nn.ReLU()
        self.res3c_branch2a = Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.bn3c_branch2a = nn.BatchNorm2d(128, eps=bn_eps, momentum=bn_mom)
        self.res3c_branch2b = Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3c_branch2b = nn.BatchNorm2d(128, eps=bn_eps, momentum=bn_mom)
        self.res3c_branch2c = Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn3c_branch2c = nn.BatchNorm2d(512, eps=bn_eps, momentum=bn_mom)

        #self.feed('res3b_relu', 
        #        'bn3c_branch2c')
        #        .add(name='res3c')
        self.res3c_relu = nn.ReLU()
        self.res3d_branch2a = Conv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.bn3d_branch2a = nn.BatchNorm2d(128, eps=bn_eps, momentum=bn_mom)
        self.res3d_branch2b = Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3d_branch2b = nn.BatchNorm2d(128, eps=bn_eps, momentum=bn_mom)
        self.res3d_branch2c = Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn3d_branch2c = nn.BatchNorm2d(512, eps=bn_eps, momentum=bn_mom)

        #self.feed('res3c_relu', 
        #        'bn3d_branch2c')
        #        .add(name='res3d')
        self.res3d_relu = nn.ReLU()
        self.res4a_branch1 = Conv2d(512, 1024, kernel_size=1, stride=2, bias=False)
        self.bn4a_branch1 = nn.BatchNorm2d(1024, eps=bn_eps, momentum=bn_mom)

        #self.feed('res3d_relu')
        self.res4a_branch2a = Conv2d(512, 256, kernel_size=1, stride=2, bias=False)
        self.bn4a_branch2a = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        self.res4a_branch2b = Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4a_branch2b = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        self.res4a_branch2c = Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn4a_branch2c = nn.BatchNorm2d(1024, eps=bn_eps, momentum=bn_mom)

        #self.feed('bn4a_branch1', 
        #        'bn4a_branch2c')
        #        .add(name='res4a')
        self.res4a_relu = nn.ReLU()
        self.res4b_branch2a = Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn4b_branch2a = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        self.res4b_branch2b = Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4b_branch2b = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        self.res4b_branch2c = Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn4b_branch2c = nn.BatchNorm2d(1024, eps=bn_eps, momentum=bn_mom)

        #self.feed('res4a_relu', 
        #        'bn4b_branch2c')
        #        .add(name='res4b')
        self.res4b_relu = nn.ReLU()
        self.res4c_branch2a = Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn4c_branch2a = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        self.res4c_branch2b = Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4c_branch2b= nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        self.res4c_branch2c = Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn4c_branch2c = nn.BatchNorm2d(1024, eps=bn_eps, momentum=bn_mom)

        #self.feed('res4b_relu', 
        #        'bn4c_branch2c')
        #        .add(name='res4c')
        self.res4c_relu = nn.ReLU()
        self.res4d_branch2a = Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn4d_branch2a = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        self.res4d_branch2b = Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4d_branch2b = nn.BatchNorm2d(256,  eps=bn_eps, momentum=bn_mom)
        self.res4d_branch2c = Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn4d_branch2c = nn.BatchNorm2d(1024,  eps=bn_eps, momentum=bn_mom)

        #self.feed('res4c_relu', 
        #        'bn4d_branch2c')
        #        .add(name='res4d')
        self.res4d_relu = nn.ReLU()
        self.res4e_branch2a = Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn4e_branch2a = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        self.res4e_branch2b = Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4e_branch2b = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        self.res4e_branch2c = Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn4e_branch2c = nn.BatchNorm2d(1024, eps=bn_eps, momentum=bn_mom)

        #self.feed('res4d_relu', 
        #        'bn4e_branch2c')
        #        .add(name='res4e')
        self.res4e_relu = nn.ReLU()
        self.res4f_branch2a = Conv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn4f_branch2a = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        self.res4f_branch2b = Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4f_branch2b = nn.BatchNorm2d(256, eps=bn_eps, momentum=bn_mom)
        self.res4f_branch2c = Conv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn4f_branch2c = nn.BatchNorm2d(1024, eps=bn_eps, momentum=bn_mom)

        #self.feed('res4e_relu', 
        #        'bn4f_branch2c')
        #        .add(name='res4f')
        self.res4f_relu = nn.ReLU()
        self.res5a_branch1 = Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False)
        self.bn5a_branch1 = nn.BatchNorm2d(2048, eps=bn_eps, momentum=bn_mom)

        #self.feed('res4f_relu')
        self.res5a_branch2a = Conv2d(1024, 512, kernel_size=1, stride=2, bias=False)
        self.bn5a_branch2a = nn.BatchNorm2d(512, eps=bn_eps, momentum=bn_mom)
        self.res5a_branch2b = Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn5a_branch2b = nn.BatchNorm2d(512, eps=bn_eps, momentum=bn_mom)
        self.res5a_branch2c = Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.bn5a_branch2c = nn.BatchNorm2d(2048, eps=bn_eps, momentum=bn_mom)

        #self.feed('bn5a_branch1', 
        #        'bn5a_branch2c')
        #        .add(name='res5a')
        self.res5a_relu = nn.ReLU()
        self.res5b_branch2a = Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.bn5b_branch2a = nn.BatchNorm2d(512, eps=bn_eps, momentum=bn_mom)
        self.res5b_branch2b = Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn5b_branch2b = nn.BatchNorm2d(512, eps=bn_eps, momentum=bn_mom)
        self.res5b_branch2c = Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.bn5b_branch2c = nn.BatchNorm2d(2048, eps=bn_eps, momentum=bn_mom)
        
        self.drop_out_d = nn.Dropout(p=0.5)
        
                    
        #self.feed('res5a_relu', 
        #        'bn5b_branch2c') 
        #        .add(name='res5b')
        self.res5b_relu = nn.ReLU()
        self.res5c_branch2a = Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.bn5c_branch2a = nn.BatchNorm2d(512, eps=bn_eps, momentum=bn_mom)
        self.res5c_branch2b = Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn5c_branch2b = nn.BatchNorm2d(512, eps=bn_eps, momentum=bn_mom)
        self.res5c_branch2c = Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.bn5c_branch2c = nn.BatchNorm2d(2048, eps=bn_eps, momentum=bn_mom)
                
        #------------------------------------------------------------------------------------     
        # decoder EDGE MAPS & CORNERS MAPS
        
        #self.feed('bn5c_branch2c') 
        self.d_2x = ConvTranspose2d(2048, 512, kernel_size=5, bias=True, stride=2, padding=0, output_padding=0)
        
        #self.feed('d_2x','res4f_relu')
        #        .concat(axis=3,name="d_concat_2x")
        self.d_4x = ConvTranspose2d(1536, 256, kernel_size=5, bias=True, stride=2, padding=0, output_padding=0)
        self.output4X_likelihood = ConvTranspose2d(256, 2, kernel_size=3, bias=True, stride=1, padding=0)
        
        #self.feed('d_4x','res3d_relu','output4X_likelihood')
        #        .concat(axis=3,name="d_concat_4x")
        self.d_8x = ConvTranspose2d(770, 128, kernel_size=5, bias=True, stride=2, padding=0, output_padding=0)
        self.output8X_likelihood = ConvTranspose2d(128, 2, kernel_size=3, bias=True, stride=1, padding=0)
        
        #self.feed('d_8x','res2c_relu','output8X_likelihood')
        #        .concat(axis=3,name="d_concat_8x")
        self.d_16x = ConvTranspose2d(386, 64, kernel_size=5, bias=True, stride=2,  padding=0, output_padding=0)
        self.output16X_likelihood = ConvTranspose2d(64, 2, kernel_size=3, bias=True, stride=1, padding=0)
        
        #self.feed('d_16x','bn_conv1','output16X_likelihood')
        #        .concat(axis=3,name="d_concat_16x")
        self.d_16x_conv1 = ConvTranspose2d(130, 64, kernel_size=3, bias=True, stride=1, padding=0)               
        self.output_likelihood = ConvTranspose2d(64, 2, kernel_size=3, bias=True, stride=1, padding=0)

    def forward(self, inputs): 
        
        outputs = {}
        #outputs["rgb_input"] = inputs
        conv1 = self.conv1(inputs)
        #outputs['conv1'] = conv1
        bn_conv1 = self.relu(self.bn_conv1(conv1))
        #outputs['bn_conv1'] = bn_conv1
        pool1 = self.pool1(bn_conv1)
        #outputs['pool1'] = pool1
        res2a_branch1 = self.res2a_branch1(pool1)
        #outputs['res2a_branch1'] = res2a_branch1
        bn2a_branch1 = self.relu(self.bn2a_branch1(res2a_branch1))
        #outputs['bn2a_branch1'] = bn2a_branch1

        res2a_branch2a = self.res2a_branch2a(pool1)
        bn2a_branch2a = self.relu(self.bn2a_branch2a(res2a_branch2a))
        res2a_branch2b = self.res2a_branch2b(bn2a_branch2a)
        bn2a_branch2b = self.relu(self.bn2a_branch2b(res2a_branch2b))
        res2a_branch2c = self.res2a_branch2c(bn2a_branch2b)
        bn2a_branch2c = self.relu(self.bn2a_branch2c(res2a_branch2c))

        res2a = torch.add(bn2a_branch1, bn2a_branch2c)
        res2a_relu = self.res2a_relu(res2a) 
        res2b_branch2a = self.res2b_branch2a(res2a_relu)
        bn2b_branch2a = self.relu(self.bn2b_branch2a(res2b_branch2a))
        res2b_branch2b = self.res2b_branch2b(bn2b_branch2a)
        bn2b_branch2b = self.relu(self.bn2b_branch2b(res2b_branch2b))
        res2b_branch2c = self.res2b_branch2c(bn2b_branch2b)
        bn2b_branch2c = self.relu(self.bn2b_branch2c(res2b_branch2c))

        res2b = torch.add(res2a_relu, bn2b_branch2c)
        res2b_relu = self.res2b_relu(res2b)
        res2c_branch2a = self.res2c_branch2a(res2b_relu)
        bn2c_branch2a = self.relu(self.bn2c_branch2a(res2c_branch2a))
        res2c_branch2b = self.res2c_branch2b(bn2c_branch2a)
        bn2c_branch2b = self.relu(self.bn2c_branch2b(res2c_branch2b))
        res2c_branch2c = self.res2c_branch2c(bn2c_branch2b)
        bn2c_branch2c = self.relu(self.bn2c_branch2c(res2c_branch2c))

        res2c = torch.add(res2b_relu, bn2c_branch2c)
        res2c_relu  = self.res2c_relu(res2c)
        #outputs['res2c_relu'] = res2c_relu
        res3a_branch1 = self.res3a_branch1(res2c_relu)
        bn3a_branch1 = self.relu(self.bn3a_branch1(res3a_branch1))

        res3a_branch2a = self.res3a_branch2a(res2c_relu)
        bn3a_branch2a = self.relu(self.bn3a_branch2a(res3a_branch2a))
        res3a_branch2b = self.res3a_branch2b(bn3a_branch2a)
        bn3a_branch2b = self.relu(self.bn3a_branch2b(res3a_branch2b))
        res3a_branch2c = self.res3a_branch2c(bn3a_branch2b)
        bn3a_branch2c = self.relu(self.bn3a_branch2c(res3a_branch2c))

        res3a = torch.add(bn3a_branch1, bn3a_branch2c) 
        res3a_relu = self.res3a_relu(res3a)
        res3b_branch2a = self.res3b_branch2a(res3a_relu)
        bn3b_branch2a = self.relu(self.bn3b_branch2a(res3b_branch2a))
        res3b_branch2b = self.res3b_branch2b(bn3b_branch2a)
        bn3b_branch2b = self.relu(self.bn3b_branch2b(res3b_branch2b))
        res3b_branch2c = self.res3b_branch2c(bn3b_branch2b)
        bn3b_branch2c = self.relu(self.bn3b_branch2c(res3b_branch2c))

        res3b = torch.add(res3a_relu, bn3b_branch2c)
        res3b_relu = self.res3b_relu(res3b)
        res3c_branch2a = self.res3c_branch2a(res3b_relu)
        bn3c_branch2a = self.relu(self.bn3c_branch2a(res3c_branch2a))
        res3c_branch2b = self.res3c_branch2b(bn3c_branch2a)
        bn3c_branch2b = self.relu(self.bn3c_branch2b(res3c_branch2b))
        res3c_branch2c = self.res3c_branch2c(bn3c_branch2b)
        bn3c_branch2c = self.relu(self.bn3c_branch2c(res3c_branch2c))

        res3c = torch.add(res3b_relu, bn3c_branch2c)
        res3c_relu = self.res3c_relu(res3c)
        res3d_branch2a = self.res3d_branch2a(res3c_relu)
        bn3d_branch2a = self.relu(self.bn3d_branch2a(res3d_branch2a))
        res3d_branch2b = self.res3d_branch2b(bn3d_branch2a)
        bn3d_branch2b = self.relu(self.bn3d_branch2b(res3d_branch2b))
        res3d_branch2c = self.res3d_branch2c(bn3d_branch2b)
        bn3d_branch2c = self.relu(self.bn3d_branch2c(res3d_branch2c))

        res3d = torch.add(res3c_relu, bn3d_branch2c)
        res3d_relu = self.res3d_relu(res3d)
        #outputs['res3d_relu'] = res3d_relu
        res4a_branch1 = self.res4a_branch1(res3d_relu)
        bn4a_branch1 = self.relu(self.bn4a_branch1(res4a_branch1))

        res4a_branch2a = self.res4a_branch2a(res3d_relu)
        bn4a_branch2a = self.relu(self.bn4a_branch2a(res4a_branch2a))
        res4a_branch2b = self.res4a_branch2b(bn4a_branch2a)
        bn4a_branch2b = self.relu(self.bn4a_branch2b(res4a_branch2b))
        res4a_branch2c = self.res4a_branch2c(bn4a_branch2b)
        bn4a_branch2c = self.relu(self.bn4a_branch2c(res4a_branch2c))

        res4a = torch.add(bn4a_branch1, bn4a_branch2c)
        #outputs['res4a'] = res4a
        res4a_relu = self.res4a_relu(res4a)
        #outputs['res4a_relu'] = res4a_relu
        res4b_branch2a = self.res4b_branch2a(res4a_relu)
        bn4b_branch2a = self.relu(self.bn4b_branch2a(res4b_branch2a))
        res4b_branch2b = self.res4b_branch2b(bn4b_branch2a)
        bn4b_branch2b = self.relu(self.bn4b_branch2b(res4b_branch2b))
        res4b_branch2c = self.res4b_branch2c(bn4b_branch2b)
        bn4b_branch2c = self.relu(self.bn4b_branch2c(res4b_branch2c))

        res4b = torch.add(res4a_relu, bn4b_branch2c)
        res4b_relu = self.res4b_relu(res4b)
        #outputs['res4b_relu'] = res4b_relu
        res4c_branch2a = self.res4c_branch2a(res4b_relu)
        #outputs['res4c_branch2a'] = res4c_branch2a
        bn4c_branch2a = self.relu(self.bn4c_branch2a(res4c_branch2a))
        #outputs['bn4c_branch2a'] = bn4c_branch2a
        res4c_branch2b = self.res4c_branch2b(bn4c_branch2a)
        #outputs['res4c_branch2b'] = res4c_branch2b
        bn4c_branch2b = self.relu(self.bn4c_branch2b(res4c_branch2b))
        #outputs['bn4c_branch2b'] = bn4c_branch2b
        res4c_branch2c = self.res4c_branch2c(bn4c_branch2b)
        #outputs['res4c_branch2c'] = res4c_branch2c
        bn4c_branch2c = self.relu(self.bn4c_branch2c(res4c_branch2c))
        #outputs['bn4c_branch2c'] = bn4c_branch2c

        res4c = torch.add(res4b_relu, bn4c_branch2c)
        #outputs['res4c'] = res4c
        res4c_relu = self.res4c_relu(res4c)
        #outputs['res4c_relu'] = res4c_relu
        res4d_branch2a = self.res4d_branch2a(res4c_relu)
        bn4d_branch2a = self.relu(self.bn4d_branch2a(res4d_branch2a))
        res4d_branch2b = self.res4d_branch2b(bn4d_branch2a)
        bn4d_branch2b = self.relu(self.bn4d_branch2b(res4d_branch2b))
        res4d_branch2c = self.res4d_branch2c(bn4d_branch2b)
        bn4d_branch2c = self.relu(self.bn4d_branch2c(res4d_branch2c))

        res4d = torch.add(res4c_relu, bn4d_branch2c)
        #outputs['res4d'] = res4d
        res4d_relu = self.res4d_relu(res4d)
        res4e_branch2a = self.res4e_branch2a(res4d_relu)
        bn4e_branch2a = self.relu(self.bn4e_branch2a(res4e_branch2a))
        res4e_branch2b = self.res4e_branch2b(bn4e_branch2a)
        bn4e_branch2b = self.relu(self.bn4e_branch2b(res4e_branch2b))
        res4e_branch2c = self.res4e_branch2c(bn4e_branch2b)
        bn4e_branch2c = self.relu(self.bn4e_branch2c(res4e_branch2c))

        res4e = torch.add(res4d_relu, bn4e_branch2c) 
        res4e_relu = self.res4e_relu(res4e)
        #outputs['res4e_relu'] = res4e_relu
        res4f_branch2a = self.res4f_branch2a(res4e_relu)
        bn4f_branch2a = self.relu(self.bn4f_branch2a(res4f_branch2a))
        res4f_branch2b = self.res4f_branch2b(bn4f_branch2a)
        bn4f_branch2b = self.relu(self.bn4f_branch2b(res4f_branch2b))
        res4f_branch2c = self.res4f_branch2c(bn4f_branch2b)
        bn4f_branch2c = self.relu(self.bn4f_branch2c(res4f_branch2c))
        #outputs['bn4f_branch2c'] = bn4f_branch2c

        res4f = torch.add(res4e_relu, bn4f_branch2c)
        #outputs['res4f'] = res4f
        res4f_relu = self.res4f_relu(res4f)
        #outputs['res4f_relu'] = res4f_relu
        res5a_branch1 = self.res5a_branch1(res4f_relu)
        bn5a_branch1 = self.relu(self.bn5a_branch1(res5a_branch1))

        res5a_branch2a = self.res5a_branch2a(res4f_relu)
        bn5a_branch2a = self.relu(self.bn5a_branch2a(res5a_branch2a))
        res5a_branch2b = self.res5a_branch2b(bn5a_branch2a)
        bn5a_branch2b = self.relu(self.bn5a_branch2b(res5a_branch2b))
        res5a_branch2c = self.res5a_branch2c(bn5a_branch2b)
        bn5a_branch2c = self.relu(self.bn5a_branch2c(res5a_branch2c))

        res5a = torch.add(bn5a_branch1, bn5a_branch2c)
        res5a_relu = self.res5a_relu(res5a)
        res5b_branch2a = self.res5b_branch2a(res5a_relu)
        bn5b_branch2a = self.relu(self.bn5b_branch2a(res5b_branch2a))
        res5b_branch2b = self.res5b_branch2b(bn5b_branch2a)
        bn5b_branch2b = self.relu(self.bn5b_branch2b(res5b_branch2b))
        res5b_branch2c = self.res5b_branch2c(bn5b_branch2b)
        bn5b_branch2c = self.relu(self.bn5b_branch2c(res5b_branch2c))
        
        res5b = torch.add(res5a_relu, bn5b_branch2c) 
        res5b_relu = self.res5b_relu(res5b)
        res5c_branch2a = self.res5c_branch2a(res5b_relu)
        bn5c_branch2a = self.drop_out_d(self.relu(self.bn5c_branch2a(res5c_branch2a)))
        res5c_branch2b = self.res5c_branch2b(bn5c_branch2a)
        bn5c_branch2b = self.drop_out_d(self.relu(self.bn5c_branch2b(res5c_branch2b)))
        res5c_branch2c = self.res5c_branch2c(bn5c_branch2b)
        bn5c_branch2c = self.relu(self.bn5c_branch2c(res5c_branch2c))
        #outputs['bn5c_branch2c'] = bn5c_branch2c
                 
        d_2x = self.d_2x(bn5c_branch2c)
        #outputs['d_2x'] = d_2x
        d_concat_2x = torch.cat((d_2x, res4f_relu), dim=1)
        #outputs['d_concat_2x'] = d_concat_2x
        d_4x = self.d_4x(d_concat_2x)
        #outputs['d_4x'] = d_4x
        output4X_likelihood = self.output4X_likelihood(d_4x)
         
        d_concat_4x = torch.cat((d_4x, res3d_relu, output4X_likelihood), dim=1)
        #outputs['d_concat_4x'] = d_concat_4x
        d_8x = self.d_8x(d_concat_4x)
        #outputs['d_8x'] = d_8x
        output8X_likelihood = self.output8X_likelihood(d_8x)
        
        d_concat_8x = torch.cat((d_8x, res2c_relu, output8X_likelihood), dim=1)
        #outputs['d_concat_8x'] = d_concat_8x
        d_16x = self.d_16x(d_concat_8x)
        #outputs[d_16x] = d_16x
        output16X_likelihood = self.output16X_likelihood(d_16x)
        
        d_concat_16x = torch.cat((d_16x, bn_conv1, output16X_likelihood), dim=1)
        #outputs['d_concat_16x'] = d_concat_16x
        d_16x_conv1 = self.d_16x_conv1(d_concat_16x)  
        #outputs['d_16x_conv1'] = d_16x_conv1             
        output_likelihood = self.output_likelihood(d_16x_conv1)
        
        
        
        
        outputs['output4X_likelihood'] = output4X_likelihood
        outputs['output8X_likelihood'] = output8X_likelihood
        outputs['output16X_likelihood'] = output16X_likelihood
        outputs['output_likelihood'] = output_likelihood

        return outputs

if __name__ == '__main__':
    input0 = torch.randn(1,3,128,256)
    model = StdConvsTFCFL()
    output0 = model(input0)
    print(output0['output_likelihood'].shape)        