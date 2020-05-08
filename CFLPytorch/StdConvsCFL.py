import torch
from torch import nn
from torch.nn import functional as F

from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params, conv_type):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect
        self._conv_type = conv_type

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=[128,256], conv_type= self._conv_type)
        StdConv2d = get_same_padding_conv2d(image_size=[128,256], conv_type= 'Std')

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = StdConv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        #if not isinstance(s,int):
        #    s = s[0]
        #print("k = ",k," s = ",s)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = StdConv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = StdConv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = StdConv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None, offset=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))     

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None, conv_type=None, layerdict=None, offsetdict=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._conv_type = conv_type
        self._offsetdict=offsetdict
        self._layerdict=layerdict

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=[128,256],conv_type=self._conv_type)
        StdConv2d = get_same_padding_conv2d(image_size=[128,256], conv_type= 'Std')
        
        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, self._conv_type))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, self._conv_type))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = StdConv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        
        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._dropout0 = nn.Dropout(p=0.3)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)


    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        skipconnection={}
        # Stem
        index = 0
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        skipconnection[index] = x

        # Blocks
        
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            index+= 1 
            x = block(x, drop_connect_rate=drop_connect_rate)
            skipconnection[index] = x
            

        # Head
        index+= 1
        #x = self._swish(self._bn1(self._conv_head(x)))
        x = self._dropout0(self._bn1(self._conv_head(x)))
        skipconnection[index] = x

        #for key in skipconnection:
        #    print("index: ",key, "shape: ", skipconnection[key].shape[2])

        return x, skipconnection


    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        #bs = inputs.size(0)
        # Convolution layers
        x, skipdict = self.extract_features(inputs)
        
        """
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        """
        return x, skipdict
   
    @classmethod
    def from_name(cls, model_name, conv_type, layerdict=None ,offsetdict=None ,override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, conv_type, layerdict ,offsetdict)

    @classmethod
    def from_pretrained(cls, model_name, conv_type, layerdict=None ,offsetdict=None, advprop=False, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, conv_type, layerdict=layerdict ,offsetdict=offsetdict, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size, conv_type=model._conv_type)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model
    
    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """ 
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


class StdConvsCFL(nn.Module):
    def __init__(self, model_name, conv_type=None, layerdict=None, offsetdict=None):
        super().__init__()
        self._encoder = EfficientNet.from_pretrained(model_name,conv_type=conv_type, layerdict=layerdict, offsetdict=offsetdict)
        self._swish = MemoryEfficientSwish()
        
        # Decoder layer
        self._upconv1a = nn.ConvTranspose2d(1280, 320, kernel_size=5, bias=True, stride=2, padding=2, output_padding=1)
        self._upconv1b = nn.ConvTranspose2d(432, 112, kernel_size=5, bias=True, stride=2, padding=2, output_padding=1)
        self._upconv1c = nn.ConvTranspose2d(112, 2, kernel_size=3, bias=True, stride=1, padding=1)
        self._upconv2a = nn.ConvTranspose2d(154, 40, kernel_size=5, bias=True, stride=2, padding=2, output_padding=1)
        self._upconv2b = nn.ConvTranspose2d(40, 2, kernel_size=3, bias=True, stride=1, padding=1)
        self._upconv3a = nn.ConvTranspose2d(66, 24, kernel_size=5, bias=True, stride=2,  padding=2, output_padding=1)
        self._upconv3b = nn.ConvTranspose2d(24, 2, kernel_size=3, bias=True, stride=1, padding=1)
        self._upconv4a = nn.ConvTranspose2d(42, 16, kernel_size=3, bias=True, stride=1, padding=1)
        self._upconv4b = nn.ConvTranspose2d(16, 2, kernel_size=3, bias=True, stride=1, padding=1)

    def forward(self, inputs):
        ret = {}
        x, skipconnection = self._encoder(inputs) 
     #------------------------------------------------------------------------------------  
        # decoder EDGE MAPS & CORNERS MAPS   
        
        d_2x = self._swish(self._upconv1a(x)) 
    
        connection = 11
        d_concat_2x = torch.cat((d_2x,skipconnection[connection]),dim=1)
        d_4x = self._swish(self._upconv1b(d_concat_2x))    
        output4x_likelihood = self._upconv1c(d_4x)    
        ret['output4x'] = output4x_likelihood

        connection = 5
        d_concat_4x = torch.cat((d_4x,skipconnection[connection],output4x_likelihood),dim=1)
        d_8x = self._swish(self._upconv2a(d_concat_4x))    
        output8x_likelihood = self._upconv2b(d_8x)
        ret['output8x'] = output8x_likelihood
        
        connection = 3
        d_concat_8x = torch.cat((d_8x,skipconnection[connection],output8x_likelihood),dim=1)
        d_16x = self._swish(self._upconv3a(d_concat_8x)) 
        output16x_likelihood = self._upconv3b(d_16x)  
        ret['output16x'] = output16x_likelihood
        
        connection = 1
        d_concat_16x = torch.cat((d_16x,skipconnection[connection],output16x_likelihood),dim=1)
        d_16x_conv1 = self._swish(self._upconv4a(d_concat_16x))
        output_likelihood = self._upconv4b(d_16x_conv1)    
        ret['output'] = output_likelihood

        return ret
    

if __name__ == '__main__':
    input0 = torch.randn(1,3,128,256)
    model = StdConvsCFL('efficientnet-b0','Std', layerdict=None, offsetdict=None)
    output0 = model(input0)
    print(output0['output'].shape)
   
 