from typing import Union
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch
from torch.nn.modules.conv import _ConvNd
# from cnnbase import ConvBase
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Convolution_calculation(input, kernel, stride):
    out_channel, in_channel, kernel_h, kernel_w = kernel.shape
    batch_size, in_channels, in_h, in_w = input.shape
    kernel_clone = kernel.clone()
    input_clone = input.clone()
    out_h = (in_h - kernel_h) // stride + 1
    out_w = (in_w  - kernel_w) // stride + 1
    kernel_clone = kernel_clone.view(out_channel, -1).transpose(0, 1)
    input_clone = F.unfold(input_clone, (kernel_h, kernel_w), stride)
    input_clone = input_clone.view(batch_size, -1, out_h * out_w).transpose(1, 2)
    output = torch.matmul(input_clone, kernel_clone)
    output = output.transpose(1, 2).reshape(batch_size, out_h, out_w, out_channel).transpose(1, 3).transpose(2,3)
    return output

import numpy as np
class Conv2d(_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def conv2d(self, input, kernel, bias = 0, stride=1, padding=0):
        torch.cuda.empty_cache()
        self.input=input
        batch_size, in_channels, in_h, in_w = input.shape
        out_channels, _, kernel_h, kernel_w = kernel.shape
        out_h = (in_h + 2 * self.padding[0] - kernel_h) // self.stride[0] + 1
        out_w = (in_w + 2 * self.padding[1] - kernel_w) // self.stride[1] + 1
        output = torch.zeros(batch_size, out_channels, out_h, out_w)
        #output=output.to(device)
        input_padded = F.pad(input, (self.padding[0], self.padding[0], self.padding[0], self.padding[0])).to(device)
        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_h):
                    for w_out in range(out_w):
                        h_in, w_in = h_out * self.stride[0], w_out * self.stride[0]
                        receptive_field = input_padded[b, :, h_in:h_in + kernel_h, w_in:w_in + kernel_w].to(device)
                        output[b, c_out, h_out, w_out] = torch.sum(receptive_field * kernel[c_out]) + bias[c_out]
        self.output = output
        torch.cuda.empty_cache()
        return output

    def forward(self, input: Tensor):
        weight = self.weight
        bias = self.bias
        return self.conv2d(input, weight, bias)

    def backward(self, ones: Tensor):
        torch.cuda.empty_cache()
        weight_rotated = torch.rot90(self.weight, 2, dims=[2, 3]).to(device)
        weight_rotated=weight_rotated.transpose(0,1).to(device)
        ones_padded = F.pad(ones, pad=[1, 1,1, 1])
        batch_size, _, in_h, in_w = ones_padded.shape
        kernel_h, kernel_w = weight_rotated.shape[2:]
        out_channels, _, _ = self.input.shape[1:]
        out_h=(in_h + 2 * self.padding[0] - kernel_h) // self.stride[0] + 1
        out_w = (in_w + 2 * self.padding[1] - kernel_w) // self.stride[1] + 1
        grad_input = torch.zeros(self.input.shape, device=self.input.device, dtype=self.input.dtype).to(device)
        grad_weight = torch.zeros(self.weight.shape, device=self.weight.device, dtype=self.weight.dtype).to(device)
        grad_bias = torch.zeros(self.bias.shape, device=self.bias.device, dtype=self.bias.dtype).to(device)
        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_h):
                    for w_out in range(out_w):
                        h_in, w_in = h_out * self.stride[0], w_out * self.stride[0]
                        grad=ones_padded[b, :, h_in:h_in + kernel_h, w_in:w_in + kernel_w].to(device)
                        grad_input[b, c_out, h_out, w_out]=torch.sum(grad*weight_rotated[c_out])
        clone_input=self.input.clone().transpose(0,1).to(device)
        clone_ones=ones.clone().transpose(0,1).to(device)
        batch_size=self.input.shape[1]
        out_channels=ones.shape[1]
        out_h=self.weight.shape[2]
        out_w=self.weight.shape[3]
        grad_weight=grad_weight.transpose(0,1)
        kernel_h=ones.shape[2]
        kernel_w=ones.shape[3]
        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_h):
                    for w_out in range(out_w):
                        h_in, w_in = h_out * self.stride[0], w_out * self.stride[0]
                        grad=clone_input[b, :, h_in:h_in + kernel_h, w_in:w_in + kernel_w].to(device)
                        grad_weight[b, c_out, h_out, w_out]=torch.sum(grad*clone_ones[c_out])
        grad_weight = grad_weight.transpose(0, 1).to(device)
        grad_bias= torch.sum(ones, dim=(0, 2, 3)).to(device)
        self.weight.grad = grad_weight
        self.bias.grad = grad_bias
        self.input.grad = grad_input
        torch.cuda.empty_cache()
        return grad_input
class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))#随机weight
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))


    def forward(self, input):
        torch.cuda.empty_cache()
        input=input.to(device)
        self.input=input
        self.output = torch.matmul(input, self.weight.t()).to(device)
        if hasattr(self, 'bias'):
            self.output += self.bias
        torch.cuda.empty_cache()
        return self.output
    def backward(self, ones: Tensor):
        torch.cuda.empty_cache()
        input_grad = torch.matmul(ones, self.weight).to(device)
        self.weight.grad = torch.matmul(ones.t(), self.input).to(device)
        if hasattr(self, 'bias'):
            self.bias.grad = ones.sum(dim=0)
        self.input.grad=input_grad
        torch.cuda.empty_cache()
        return self.input.grad
class CrossEntropyLoss():
    def __init__(self):
        pass
    def __call__(self, input, target):
        self.input = input
        self.target = target
        loss = 0
        batch_size = input.shape[0]
        for i in range(batch_size):
            log_softmax = input[i] - torch.log(torch.sum(torch.exp(input[i])))
            loss += -log_softmax[target[i]]
        self.output = loss / batch_size
        return self.output
    def backward(self):
        self.input.grad = torch.zeros_like(self.input)
        batch_size = self.input.shape[0]
        for i in range(batch_size):
            exp_input = torch.exp(self.input[i])
            softmax = exp_input / torch.sum(exp_input)
            softmax[self.target[i]] -= 1
            self.input.grad[i] = softmax
        self.input.grad /= batch_size
        return self.input.grad
