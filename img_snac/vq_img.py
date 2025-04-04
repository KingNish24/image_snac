# image_snac/img_snac/vq_img.py
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layers_img import WNConv2d


class VectorQuantize2d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        stride: Tuple[int, int] = (1, 1), # Spatial stride (sh, sw)
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.stride_h, self.stride_w = stride
        self.commitment_weight = commitment_weight

        self.in_proj = WNConv2d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv2d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z (torch.Tensor): Input tensor (B, C, H, W).
        Returns:
            torch.Tensor: Quantized output tensor (B, C, H_out, W_out), where H_out, W_out depend on stride.
            torch.Tensor: Codebook indices (B, H_pooled, W_pooled).
            torch.Tensor: VQ loss (scalar).
        """
        B, C, H, W = z.shape

        # Apply spatial pooling if stride > 1
        if self.stride_h > 1 or self.stride_w > 1:
            # Use AvgPool2d for downsampling before quantization
            z_pooled = F.avg_pool2d(z, kernel_size=(self.stride_h, self.stride_w), stride=(self.stride_h, self.stride_w))
        else:
            z_pooled = z

        # Project pooled features
        z_e = self.in_proj(z_pooled)  # z_e: (B, D_codebook, H_pooled, W_pooled)
        D_codebook = z_e.shape[1]
        H_pooled, W_pooled = z_e.shape[2], z_e.shape[3]

        # Flatten spatial dimensions for codebook lookup
        z_e_flat = rearrange(z_e, "b d h w -> (b h w) d")

        # --- Codebook Lookup ---
        # L2 normalize for cosine similarity like ViT-VQGAN (optional but common)
        # z_e_flat_norm = F.normalize(z_e_flat, dim=1)
        # codebook_norm = F.normalize(self.codebook.weight, dim=1)
        # distances = torch.cdist(z_e_flat_norm, codebook_norm, p=2).pow(2) # Squared L2

        # Standard Euclidean distance (closer to original VQ-VAE)
        distances = (
            z_e_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_e_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1, keepdim=True).t()
        ) # Shape: (B*H_pooled*W_pooled, N_codebook)

        # Find nearest neighbors
        indices_flat = torch.argmin(distances, dim=1) # (B*H_pooled*W_pooled)
        indices = rearrange(indices_flat, "(b h w) -> b h w", b=B, h=H_pooled, w=W_pooled)

        # Retrieve quantized vectors
        z_q_flat = F.embedding(indices_flat, self.codebook.weight) # (B*H_pooled*W_pooled, D_codebook)
        z_q = rearrange(z_q_flat, "(b h w) d -> b d h w", b=B, h=H_pooled, w=W_pooled)

        # --- Loss Calculation ---
        # Commitment loss: encourages encoder outputs to be close to codebook vectors
        commitment_loss = F.mse_loss(z_e.detach(), z_q)
        # Codebook loss: encourages codebook vectors to be close to encoder outputs (implicitly done by optimizer)
        # Can add an explicit codebook loss term if desired (e.g., VQ-VAE paper)
        # vq_loss = commitment_loss * self.commitment_weight # Simple version

        # Add codebook loss component (from original VQ-VAE paper / Sonnet implementation)
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator: Copy gradients from z_q to z_e
        z_q = z_e + (z_q - z_e).detach()

        # Project back to original feature dimension
        z_q = self.out_proj(z_q) # (B, C, H_pooled, W_pooled)

        # Upsample quantized features to match original pooled feature map size if needed
        # This version assumes subsequent layers handle the upsampling, or residual is added *before* upsampling.
        # For SNAC-like residual VQ, we need to match the *input* size `z` for the residual connection.

        # If using strides, upsample z_q to match the input `z` shape H, W for residual connection in RVQ
        if self.stride_h > 1 or self.stride_w > 1:
             z_q = F.interpolate(z_q, size=(H, W), mode='nearest')


        return z_q, indices, vq_loss


class ResidualVectorQuantize2d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        codebook_size: int = 8192, # As requested
        codebook_dim: int = 16, # Dimension of each codebook vector
        n_codebooks: int = 4, # Number of VQ stages
        vq_strides: Optional[List[Tuple[int, int]]] = None, # Spatial strides per quantizer e.g., [(4,4), (2,2), (1,1), (1,1)]
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        if vq_strides is None:
            # Default: No stride (all quantizers see full resolution features)
            vq_strides = [(1, 1)] * n_codebooks
        else:
            assert len(vq_strides) == n_codebooks, "Number of strides must match number of codebooks"
        self.vq_strides = vq_strides

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize2d(input_dim, codebook_size, codebook_dim, stride, commitment_weight)
                for stride in vq_strides
            ]
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Args:
            z (torch.Tensor): Input features from encoder (B, C, H, W).
        Returns:
            torch.Tensor: Final quantized tensor (B, C, H, W).
            List[torch.Tensor]: List of codebook indices from each quantizer [(B, H_i, W_i), ...].
            torch.Tensor: Total VQ loss (summed across quantizers).
        """
        z_q_total = 0.0
        residual = z
        codes = []
        total_vq_loss = 0.0

        for i, quantizer in enumerate(self.quantizers):
            # Pass the *current residual* to the quantizer
            z_q_i, indices_i, vq_loss_i = quantizer(residual)

            # Add the quantized output to the total
            z_q_total = z_q_total + z_q_i

            # Update the residual for the next stage
            residual = residual - z_q_i # Subtract the quantized part

            codes.append(indices_i)
            total_vq_loss = total_vq_loss + vq_loss_i

        # Return the sum of quantized outputs, the list of codes, and average loss
        return z_q_total, codes, total_vq_loss / self.n_codebooks

    def from_codes(self, codes: List[torch.Tensor]) -> torch.Tensor:
        """
        Reconstructs the quantized feature map from codebook indices.

        Args:
            codes (List[torch.Tensor]): List of codebook indices [(B, H_i, W_i), ...].
                                       Indices must match the shapes produced by the forward pass.
        Returns:
            torch.Tensor: Reconstructed quantized feature map (B, C, H, W).
        """
        assert len(codes) == self.n_codebooks
        z_q_total = 0.0
        target_h, target_w = -1, -1 # Determine target size from the first code map that wasn't strided by > 1

        # Pass 1: Decode codes and project, find target H, W
        decoded_projected = []
        for i in range(self.n_codebooks):
            quantizer = self.quantizers[i]
            indices = codes[i] # (B, H_i, W_i)

            # Decode indices to codebook vectors
            z_p_i_flat = F.embedding(indices.view(-1), quantizer.codebook.weight) # (B*H_i*W_i, D_codebook)
            z_p_i = rearrange(z_p_i_flat, "(b h w) d -> b d h w", b=indices.shape[0], h=indices.shape[1], w=indices.shape[2])

            # Project back to input dimension
            z_q_i = quantizer.out_proj(z_p_i) # (B, C, H_i, W_i)
            decoded_projected.append(z_q_i)

            # Infer the target H, W (largest H, W among the decoded maps)
            if quantizer.stride_h == 1 and quantizer.stride_w == 1:
                 if target_h == -1:
                     target_h, target_w = z_q_i.shape[2], z_q_i.shape[3]
            elif target_h == -1: # Fallback if all have strides > 1 (use the least strided one)
                 current_total_stride = quantizer.stride_h * quantizer.stride_w
                 if i == 0 or current_total_stride < min_stride:
                     min_stride = current_total_stride
                     target_h = z_q_i.shape[2] * quantizer.stride_h
                     target_w = z_q_i.shape[3] * quantizer.stride_w

        if target_h == -1 or target_w == -1:
             raise ValueError("Could not determine target H, W for decoding. Ensure at least one vq_stride is (1,1) or provide target_shape.")


        # Pass 2: Upsample each decoded map to target H, W and sum
        for i in range(self.n_codebooks):
             z_q_i = decoded_projected[i]
             if z_q_i.shape[2] != target_h or z_q_i.shape[3] != target_w:
                 z_q_i = F.interpolate(z_q_i, size=(target_h, target_w), mode='nearest')
             z_q_total += z_q_i

        return z_q_total
