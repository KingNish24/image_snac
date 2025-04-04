# image_snac/img_snac/attention2d.py
import torch
from einops import rearrange
from torch import nn

# Note: 2D Rotary Positional Embeddings are more complex than 1D.
# For simplicity, this adaptation uses standard LayerNorm + MHA without RoPE.
# A full RoPE2D implementation would require careful handling of H and W dimensions.
# Alternatively, fixed 2D positional encodings (like in ViT) could be added.

class LocalMHA2d(nn.Module):
    def __init__(self, dim=256, window_size=(8, 8), dim_head=64):
        """
        Local Multi-Head Self-Attention for 2D feature maps.

        Args:
            dim (int): Feature dimension.
            window_size (tuple): Height and width of the attention window (e.g., (8, 8)).
            dim_head (int): Dimension of each attention head.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.heads = dim // dim_head
        self.window_h, self.window_w = window_size
        assert dim % dim_head == 0, "dim must be divisible by dim_head"

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        # Precompute relative offsets for indexing within windows later, if needed
        # For scaled_dot_product_attention, explicit relative embeddings are less direct

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        assert C == self.norm.normalized_shape[0], f"Input channel {C} doesn't match LayerNorm dim {self.norm.normalized_shape[0]}"
        residual = x

        # Normalize features across the channel dimension
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.norm(x)

        # Pad H and W to be divisible by window size
        pad_h = (self.window_h - H % self.window_h) % self.window_h
        pad_w = (self.window_w - W % self.window_w) % self.window_w
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h)) # Pad C, W, H dims

        padded_H, padded_W = x.shape[1], x.shape[2]
        num_windows_h = padded_H // self.window_h
        num_windows_w = padded_W // self.window_w

        # Project to Q, K, V
        q, k, v = self.to_qkv(x).chunk(3, dim=-1) # (B, H, W, C*3) -> 3 * (B, H, W, C)

        # Rearrange into windows and heads
        # (B, H, W, C) -> (B, num_win_h, win_h, num_win_w, win_w, num_heads * head_dim)
        q, k, v = map(
            lambda t: rearrange(t, 'b (nh wh) (nw ww) (h d) -> (b nh nw) h (wh ww) d',
                                wh=self.window_h, ww=self.window_w, h=self.heads),
            (q, k, v)
        )
        # Shape: (B * num_windows, num_heads, window_h * window_w, head_dim)

        # Scaled dot-product attention within each window
        # Requires PyTorch 2.0+
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # Rearrange back to original shape
        # (B * num_windows, num_heads, window_h * window_w, head_dim) -> (B, H, W, C)
        out = rearrange(out, '(b nh nw) h (wh ww) d -> b (nh wh) (nw ww) (h d)',
                        nh=num_windows_h, nw=num_windows_w, wh=self.window_h, ww=self.window_w, h=self.heads)

        # Project out
        out = self.to_out(out)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :H, :W, :]

        # Permute back and add residual
        out = out.permute(0, 3, 1, 2) # (B, C, H, W)

        return out + residual
