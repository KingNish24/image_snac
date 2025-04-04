# image_snac/img_snac/layers_img.py
import math
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from .attention2d import LocalMHA2d

# --- Weight Norm Convolution Wrappers ---
def WNConv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))

def WNConvTranspose2d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose2d(*args, **kwargs))

# --- Activation Function (Optional: Snake or use standard like GELU/SiLU) ---
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1) # Flatten spatial dims
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape) # Reshape back
    return x

class Snake2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Shared alpha across spatial dimensions for simplicity
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        # Using snake activation, you can replace with nn.SiLU() or nn.GELU() for simplicity
        # return snake(x, self.alpha)
        return torch.nn.functional.silu(x) # Using SiLU as a robust default


# --- Residual Unit for 2D ---
class ResidualUnit2d(nn.Module):
    def __init__(self, dim=16, kernel_size=3, dilation=1, use_snake=False):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        Act = Snake2d if use_snake else nn.SiLU # Or nn.GELU

        self.block = nn.Sequential(
            Act(dim),
            WNConv2d(dim, dim, kernel_size=kernel_size, dilation=dilation, padding=pad),
            Act(dim),
            WNConv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        # Ensure output spatial dims match input if padding/dilation causes issues (unlikely with 'same' padding)
        # Simple residual connection:
        return x + y


# --- Encoder Block for 2D ---
class EncoderBlock2d(nn.Module):
    def __init__(self, input_dim, output_dim, stride=2, use_snake=False):
        super().__init__()
        kernel_size = 2 * stride # Kernel size related to stride for downsampling conv

        self.block = nn.Sequential(
            ResidualUnit2d(input_dim, kernel_size=3, dilation=1, use_snake=use_snake),
            ResidualUnit2d(input_dim, kernel_size=3, dilation=3, use_snake=use_snake), # Example dilations
            ResidualUnit2d(input_dim, kernel_size=3, dilation=9, use_snake=use_snake),
            (Snake2d(input_dim) if use_snake else nn.SiLU()),
            WNConv2d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=math.ceil((kernel_size - stride) / 2), # Calculate padding for strided conv
            ),
        )

    def forward(self, x):
        return self.block(x)


# --- Decoder Block for 2D ---
class DecoderBlock2d(nn.Module):
    def __init__(self, input_dim, output_dim, stride=2, use_snake=False):
        super().__init__()
        kernel_size = 2 * stride # Kernel size related to stride for upsampling conv

        layers = [
            (Snake2d(input_dim) if use_snake else nn.SiLU()),
            WNConvTranspose2d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=math.ceil((kernel_size - stride) / 2), # Padding for transpose conv
                output_padding=stride % 2, # Needed if stride is odd to match output shape
            ),
            ResidualUnit2d(output_dim, kernel_size=3, dilation=1, use_snake=use_snake),
            ResidualUnit2d(output_dim, kernel_size=3, dilation=3, use_snake=use_snake), # Example dilations
            ResidualUnit2d(output_dim, kernel_size=3, dilation=9, use_snake=use_snake),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# --- Main Encoder and Decoder Modules ---
class Encoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        d_model=64,
        strides=[2, 2, 2, 2], # Example strides for downsampling
        attn_window_size=None, # e.g., (8, 8)
        use_snake=False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.d_model = d_model
        self.strides = strides

        layers = [WNConv2d(input_channels, d_model, kernel_size=7, padding=3)]
        current_dim = d_model
        for stride in strides:
            output_dim = current_dim * 2
            layers.append(EncoderBlock2d(current_dim, output_dim, stride=stride, use_snake=use_snake))
            current_dim = output_dim

        if attn_window_size is not None and isinstance(attn_window_size, tuple):
            layers.append(LocalMHA2d(dim=current_dim, window_size=attn_window_size))

        # Final convolution
        layers.append(WNConv2d(current_dim, current_dim, kernel_size=3, padding=1))
        self.block = nn.Sequential(*layers)
        self.output_dim = current_dim
        self.downsampling_factor = int(math.prod(strides))

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim, # Should match Encoder output_dim
        output_channels=3,
        d_model=1024, # Base dim for decoder (can be different from encoder)
        strides=[2, 2, 2, 2], # Should reverse the encoder strides
        attn_window_size=None, # e.g., (8, 8)
        use_snake=False,
    ):
        super().__init__()
        self.output_channels = output_channels

        layers = [WNConv2d(input_dim, d_model, kernel_size=3, padding=1)]

        if attn_window_size is not None and isinstance(attn_window_size, tuple):
            layers.append(LocalMHA2d(dim=d_model, window_size=attn_window_size))

        current_dim = d_model
        for i, stride in enumerate(reversed(strides)): # Reverse strides for upsampling
            output_dim = current_dim // 2 if i < len(strides) - 1 else d_model // (2**(len(strides)-1)) # Rough dim calculation
            # Ensure output_dim doesn't go below a minimum, e.g., 32
            output_dim = max(32, output_dim)
            layers.append(DecoderBlock2d(current_dim, output_dim, stride=stride, use_snake=use_snake))
            current_dim = output_dim

        layers.append(Snake2d(current_dim) if use_snake else nn.SiLU())
        layers.append(WNConv2d(current_dim, output_channels, kernel_size=7, padding=3))
        layers.append(nn.Sigmoid()) # Output in [0, 1] range for images

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
