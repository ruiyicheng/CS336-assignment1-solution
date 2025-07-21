import torch
from einops import rearrange, einsum
# A = torch.randn(2, 3, 4, 5)
# B = torch.randn(5,6)
# C = einsum(A,B,"batch i j k, k l -> batch i j l")
# print(C.shape)
# images = torch.randn(64, 128, 128, 3)
# dim_by = torch.linspace(start = 0.0,  end = 1.0, steps = 10)
# dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
# image_rearr = rearrange(images, "b h w c -> b 1 h w c")
# dimmed_images = dim_value * image_rearr
# print(dimmed_images.shape)

# dimmed_images = einsum(images, dim_by, "b h w c, dim_value -> b dim_value h w c")
channels_last = torch.randn(64,32,32,3)
B = torch.randn(32*32,32*32)
# channels_last_flat = channels_last.view(-1,channels_last.size(1)*channels_last.size(2),channels_last.size(3))
# channels_first_flat = channels_last_flat.transpose(1,2)
# channel_first_flat_transformed = channels_first_flat @ B.T
# channel_last_flat_transformed = channel_first_flat_transformed.transpose(1,2)
# channels_last_transformed = channel_last_flat_transformed.view(*channels_last.shape)
# print(channels_last_transformed.shape)

height = width = 32
channels_first = rearrange(channels_last,"b h w c -> b c (h w)")
channels_first_transformed = einsum(channels_first,B,"b c input, output input -> b c output")
channels_last_transformed = rearrange(channels_first,"b c (h w) -> b h w c",h=32)
print(channels_last_transformed.shape)

