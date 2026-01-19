import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

# Basic Convolution Layer
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# Cubic Attention Layer
class cubic_attention(nn.Module):
    def __init__(self, dim, group, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta

# Residual Block with Optional Cubic Attention
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        if filter:
            self.cubic_11 = cubic_attention(in_channel // 2, group=1, kernel=11)
            self.cubic_7 = cubic_attention(in_channel // 2, group=1, kernel=7)
        self.filter = filter

    def forward(self, x):
        out = self.conv1(x)
        if self.filter:
            out = torch.chunk(out, 2, dim=1)
            out_11 = self.cubic_11(out[0])
            out_7 = self.cubic_7(out[1])
            out = torch.cat((out_11, out_7), dim=1)
        out = self.conv2(out)
        return out + x

# Spatial Strip Attention with SLSH
class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel // 2, 1) if H else (1, kernel // 2)
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size=1, stride=1, bias=False)
        self.filter_act = nn.Sigmoid()
        self.slsh = SLSH(dim)

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.shape
        spatial_size = h * w
        
        # Step 2: Apply SLSH with sort_and_split
        x_flat = x.view(n, c, -1).permute(0, 2, 1)  # [n, h*w, c]
        
        hash_codes = self.slsh.compute_hashes(x_flat)
        attention_groups = self.slsh.sort_and_split(x_flat, hash_codes, spatial_size)
        
        best_group = self.select_best_group(attention_groups)
        
        best_group = best_group.permute(0, 2, 1)  # [n, c, spatial_dim]
        best_group = best_group.view(n, c, h, w)  # [n, c, h, w]
        
        filter = self.conv(best_group)
        
        padded_x = self.pad(x)
        unfolded_x = F.unfold(padded_x, kernel_size=self.kernel)
        x_reshaped = unfolded_x.view(n, self.group, c // self.group, self.k, h * w)
        
        filter = filter.view(n, -1, self.k, h * w).unsqueeze(2)
        filter = self.filter_act(filter)
        
        out = torch.sum(x_reshaped * filter, dim=3).view(n, c, h, w)
        return out

    # Updated select_best_group function
    def select_best_group(self, attention_groups):
        stacked_groups = torch.stack(attention_groups)  # [num_groups, batch_size, channels, spatial_dim]
        group_norms = stacked_groups.norm(p=2, dim=(2, 3)).sum(dim=1)  # Sum L2 norms across channels and spatial dimensions
        best_group_idx = torch.argmax(group_norms)
        return attention_groups[best_group_idx]

# SLSH with Sort and Split
class SLSH:
    def __init__(self, input_dim, num_hashes=10):
        self.input_dim = input_dim
        self.num_hashes = num_hashes
        self.random_vectors = torch.randn(num_hashes, input_dim).float()
        
    def compute_hashes(self, input_tensor):
        input_tensor = input_tensor.float()
        random_vectors = self.random_vectors.to(input_tensor.device)
        hash_codes = (projections > 0).float()  # Binary hash codes
        return hash_codes

    def sort_and_split(self, input_tensor, hash_codes, spatial_size, num_groups=5):
        batch_size = input_tensor.size(0)
        powers_of_two = torch.pow(2, torch.arange(self.num_hashes, device=hash_codes.device, dtype=torch.float))
        
        sorted_tensors = []
        for i in range(batch_size):
            indices = torch.argsort(decimal_codes[i])
            sorted_tensor = input_tensor[i][indices]

            # Ensure sorted_tensor has exactly spatial_size by padding if needed
            if sorted_tensor.size(0) > spatial_size:
                sorted_tensor = sorted_tensor[:spatial_size]
            elif sorted_tensor.size(0) < spatial_size:
                pad_size = spatial_size - sorted_tensor.size(0)
                padding = sorted_tensor[-1:].repeat(pad_size, 1)
                sorted_tensor = torch.cat([sorted_tensor, padding], dim=0)
            
            sorted_tensors.append(sorted_tensor)

        return sorted_tensors
