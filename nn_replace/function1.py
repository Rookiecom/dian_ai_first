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