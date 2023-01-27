'''Adapted from the transformer_net.py example on the Pytorch github repo.

Link:
https://github.com/pytorch/examples/blob/0c1654d6913f77f09c0505fb284d977d89c17c1a/fast_neural_style/neural_style/transformer_net.py
'''
import torch
import torch.nn as nn
from torch.nn import functional as F, init
import dotmap
import random

ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
}

class Viewmaker(torch.nn.Module):
    '''Network that maps an image and noise vector to another image of the same size.'''
    def __init__(self, filter_size, noise_dim, device, num_channels=3, L1_forced=False,
                 bound_magnitude=0.05, divmaker=False, activation='relu', clamp=True, symmetric_clamp=False, num_res_blocks=5):
        super().__init__()

        self.num_channels = num_channels
        self.filter_size = filter_size
        self.num_res_blocks = num_res_blocks 

        self.activation = activation
        self.clamp = clamp
        self.symmetric_clamp = symmetric_clamp
        
        # Initial convolution layers (+ 1 for noise filter)
        self.conv1 = ConvLayer(self.num_channels + 1, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        
        # Residual layers +N for added random channels
        self.res1 = ResidualBlock(128 + 1)
        self.res2 = ResidualBlock(128 + 2)
        self.res3 = ResidualBlock(128 + 3)
        self.res4 = ResidualBlock(128 + 4)
        self.res5 = ResidualBlock(128 + 5)
        
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(
            128 + self.num_res_blocks, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(
            64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        # 3 channels for warping 
        self.deconv3 = ConvLayer(32, self.num_channels, kernel_size=9, stride=1)
    
        # Non-linearities
        self.act = ACTIVATIONS[self.activation]()
        self.L1_forced = L1_forced
        self.bound_magnitude = bound_magnitude
        self.divmaker = divmaker
 
        print(f"Set up viewmaker model with bound magnitude: {self.bound_magnitude}, divmaker: {self.divmaker}")


    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        filter_size = x.size(-1)
        shp = (batch_size, num, filter_size, filter_size)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1, 1)
        return torch.cat((x, noise), dim=1)

    def basic_net(self, y, num_res_blocks=5, bound_multiplier=1):
        if num_res_blocks not in list(range(6)):
            raise ValueError(f'num_res_blocks must be in {list(range(6))}, got {num_res_blocks}.')

        y = self.add_noise_channel(y, bound_multiplier=bound_multiplier)
        y = self.act(self.in1(self.conv1(y)))
        y = self.act(self.in2(self.conv2(y)))
        y = self.act(self.in3(self.conv3(y)))

        # [batch_size, 128]
        features = y.clone().mean([-1, -2])
        
        for i, res in enumerate([self.res1, self.res2, self.res3, self.res4, self.res5]):
            if i < num_res_blocks:
                y = res(self.add_noise_channel(y, bound_multiplier=bound_multiplier))

        y = self.act(self.in4(self.deconv1(y)))
        y = self.act(self.in5(self.deconv2(y)))
        y = self.deconv3(y)

        return y, features
    
    def smaller_net(self, y):
        y = self.add_noise_channel(y)
        y = self.act(self.sm_in1(self.sm_conv1(y)))

        # [batch_size, 32]
        features = y.clone().mean([-1, -2])

        y = self.sm_res1(self.add_noise_channel(y))

        # y = self.act(self.sm_in2(self.sm_deconv1(y)))
        y = self.sm_deconv2(y)

        return y, features

    def add_smaller_params(self):
        self.sm_conv1 = ConvLayer(self.num_channels + 1, 8, kernel_size=3, stride=1)
        self.sm_in1 = torch.nn.InstanceNorm2d(8)
        self.sm_res1 = ResidualBlock(8+1)
        # self.sm_deconv1 = ConvLayer(32 + 1, 32, kernel_size=3, stride=1)
        # self.sm_in2 = torch.nn.InstanceNorm2d(32)
        self.sm_deconv2 = ConvLayer(8 + 1, self.num_channels, kernel_size=3, stride=1)

    def add_batch_norm_params(self):
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.bn5 = torch.nn.BatchNorm2d(32)

        self.bn_res1 = torch.nn.BatchNorm2d(128+1)
        self.bn_res2 = torch.nn.BatchNorm2d(128+2)
        self.bn_res3 = torch.nn.BatchNorm2d(128+3)
        self.bn_res4 = torch.nn.BatchNorm2d(128+4)
        self.bn_res5 = torch.nn.BatchNorm2d(128+5)

    def batchnorm_net(self, y):
        y = self.add_noise_channel(y)
        y = self.act(self.bn1(self.conv1(y)))
        y = self.act(self.bn2(self.conv2(y)))
        y = self.act(self.bn3(self.conv3(y)))

        # [batch_size, 128]
        features = y.clone().mean([-1, -2])
        
        y = self.bn_res1(self.res1(self.add_noise_channel(y)))
        y = self.bn_res2(self.res2(self.add_noise_channel(y)))
        y = self.bn_res3(self.res3(self.add_noise_channel(y)))
        y = self.bn_res4(self.res4(self.add_noise_channel(y)))
        y = self.bn_res5(self.res5(self.add_noise_channel(y)))

        y = self.act(self.bn4(self.deconv1(y)))
        y = self.act(self.bn5(self.deconv2(y)))
        y = self.deconv3(y)

        return y, features
    
    def apply_kernel(self, x, kernel):

        kernel = kernel / kernel.norm(dim=-1, keepdim=True) # Enforce common norm.
        kernel = kernel.view(-1, 1, 3, 3).repeat(1, self.num_channels, 1, 1)  # Repeat for channels.
        # x has size [batch_size, C, W, H]
        # kernel has size [batch_size, C, W, H]
        return F.conv2d(x, kernel, groups=x.size(0))

    def get_delta(self, y_pixels, bound_multiplier=1):
        '''Produces constrained perturbation.'''
        bound_magnitude = self.bound_magnitude
        if self.divmaker:
            delta = torch.tanh(y_pixels)  # Project to [-1, 1]
            # Scale all deltas down
            avg_magnitude = delta.abs().mean([1, 2, 3], keepdim=True)
            max_magnitude = bound_magnitude
            delta = delta * max_magnitude / (avg_magnitude + 1e-4)
            return delta

        if self.L1_forced:
            delta = torch.tanh(y_pixels)
            avg_magnitude = delta.abs().mean([1,2,3], keepdim=True)
            # scale down average change if too big.
            max_magnitude = bound_magnitude
            delta = delta * max_magnitude / (avg_magnitude + 1e-4)
        else:
            raise ValueError('Viewmaker constraint not specified')
        return delta

    def forward(self, x):
        bound_multiplier = 1
        y = x
        y, features = self.basic_net(y, self.num_res_blocks, bound_multiplier)
        y_pixels = y[:, :self.num_channels]  # remove displacement field component if extant.

        delta = self.get_delta(y_pixels, bound_multiplier)
        result = x + delta

        if self.clamp and not self.symmetric_clamp:
            result = torch.clamp(result, 0, 1.0)
        elif self.clamp and self.symmetric_clamp:
            result = torch.clamp(result, -1.0, 1.0)

        return result


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        residual = x
        out = self.act(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(
                x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
