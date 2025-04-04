# image_snac/img_snac/model_img.py
import json
import math
import os
from typing import List, Tuple, Optional, Dict

import torch
from torch import nn
import numpy as np

from .layers_img import Encoder, Decoder
from .vq_img import ResidualVectorQuantize2d


class ImageVQVAE(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        encoder_config: Dict = { # Default reasonable config
            "d_model": 64,
            "strides": [2, 2, 2, 2], # Total downsampling 16x
            "attn_window_size": (8, 8),
        },
        decoder_config: Dict = { # Should roughly mirror encoder
             "d_model": 1024, # Can be different, often larger
             "strides": [2, 2, 2, 2], # Must match encoder strides reversed
             "attn_window_size": (8, 8),
        },
        vq_config: Dict = {
            "codebook_size": 8192, # User requested
            "codebook_dim": 16,    # Dimension of each code vector
            "n_codebooks": 4,      # Number of residual VQ stages
            "vq_strides": [(4,4), (2,2), (1,1), (1,1)], # Example spatial strides for hierarchy
            "commitment_weight": 0.25,
        },
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.vq_config = vq_config

        # --- Instantiate Components ---
        self.encoder = Encoder(input_channels=input_channels, **encoder_config)

        # Ensure decoder strides match encoder strides reversed
        assert list(reversed(encoder_config.get("strides", []))) == decoder_config.get("strides", []), \
               "Decoder strides must be the reverse of encoder strides"

        # Calculate latent dim from encoder output
        latent_dim = self.encoder.output_dim
        self.latent_dim = latent_dim

        # Calculate total downsampling factor required by encoder AND VQ strides
        self.encoder_downsampling = self.encoder.downsampling_factor
        vq_strides_flat = [s for stride_pair in vq_config.get("vq_strides", [(1,1)]*vq_config.get("n_codebooks",1)) for s in stride_pair]
        self.max_vq_stride = max(vq_strides_flat) if vq_strides_flat else 1
        # Total factor needs to accommodate encoder AND max VQ stride relative to encoder output
        self.total_downsampling_factor = self.encoder_downsampling * self.max_vq_stride # Check this logic carefully
        # Simpler: Pad to encoder downsampling * Attention window? VQ happens on latent space.
        # Let's pad based on encoder downsampling and attention window size if used.
        attn_win_h, attn_win_w = encoder_config.get("attn_window_size", (1,1)) or (1,1)
        self.pad_factor_h = math.lcm(self.encoder_downsampling, attn_win_h)
        self.pad_factor_w = math.lcm(self.encoder_downsampling, attn_win_w)


        self.quantizer = ResidualVectorQuantize2d(input_dim=latent_dim, **vq_config)

        self.decoder = Decoder(
            input_dim=latent_dim, # Decoder input is RVQ output dim
            output_channels=output_channels,
            **decoder_config
        )


    def _calculate_padding(self, H: int, W: int) -> Tuple[int, int, int, int]:
        """Calculates padding needed for height and width."""
        pad_h = (self.pad_factor_h - H % self.pad_factor_h) % self.pad_factor_h
        pad_w = (self.pad_factor_w - W % self.pad_factor_w) % self.pad_factor_w
        # Pad left, right, top, bottom - F.pad wants (left, right, top, bottom)
        # We pad symmetrically for simplicity here (pad right and bottom)
        return 0, pad_w, 0, pad_h # (pad_left, pad_right, pad_top, pad_bottom)

    def preprocess(self, img_data: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pads image data to be compatible with network strides."""
        B, C, H, W = img_data.shape
        padding = self._calculate_padding(H, W)

        if any(p > 0 for p in padding):
            img_data = nn.functional.pad(img_data, padding, mode='replicate') # Use replicate or reflect padding

        return img_data, (H, W) # Return padded image and original size

    def forward(self, img_data: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Encodes and decodes the image.

        Args:
            img_data (torch.Tensor): Input image tensor (B, C, H, W). Values in [0, 1].
        Returns:
            torch.Tensor: Reconstructed image tensor (B, C, H, W).
            List[torch.Tensor]: List of codebook indices from each quantizer.
            torch.Tensor: VQ loss.
        """
        original_size = img_data.shape[2:] # Store H, W
        img_padded, _ = self.preprocess(img_data)

        # Encode
        z = self.encoder(img_padded) # (B, D_latent, H', W')

        # Quantize
        z_q, codes, vq_loss = self.quantizer(z) # z_q: (B, D_latent, H', W'), codes: List[(B, H_i, W_i)]

        # Decode
        img_hat_padded = self.decoder(z_q) # (B, C, H_padded, W_padded)

        # Crop back to original size
        H_orig, W_orig = original_size
        img_hat = img_hat_padded[..., :H_orig, :W_orig] # Crop using Ellipsis

        return img_hat, codes, vq_loss

    def encode(self, img_data: torch.Tensor) -> List[torch.Tensor]:
        """Encodes image data to discrete codes."""
        original_size = img_data.shape[2:]
        img_padded, _ = self.preprocess(img_data)
        z = self.encoder(img_padded)
        _, codes, _ = self.quantizer(z) # Ignore z_q and vq_loss
        # TODO: Optionally store original size H, W alongside codes if needed for decoding later
        return codes

    def decode(self, codes: List[torch.Tensor], target_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Decodes discrete codes back to an image."""
        z_q = self.quantizer.from_codes(codes)
        # Note: z_q will have the spatial dimensions H', W' determined by the encoder output
        # and VQ upsampling.

        img_hat_padded = self.decoder(z_q)

        # Crop if target_shape is provided
        if target_shape:
            H_orig, W_orig = target_shape
            # Need to be careful: H_padded/W_padded might be smaller than H_orig/W_orig if padding was 0
            # This happens if input was already divisible by pad_factor
            H_padded, W_padded = img_hat_padded.shape[2:]
            img_hat = img_hat_padded[..., :min(H_orig, H_padded), :min(W_orig, W_padded)]
        else:
            # If no target shape, return the full decoded output (potentially padded)
            img_hat = img_hat_padded
            print("Warning: Decoding without target_shape. Output size might include padding.")

        return img_hat

    @classmethod
    def from_config(cls, config_path):
        """Loads model from a JSON config file."""
        with open(config_path, "r") as f:
            config = json.load(f)
        # Could add more validation here
        model = cls(
             input_channels=config.get("input_channels", 3),
             output_channels=config.get("output_channels", 3),
             encoder_config=config.get("encoder_config", {}),
             decoder_config=config.get("decoder_config", {}),
             vq_config=config.get("vq_config", {}),
        )
        return model

    def save_config(self, config_path):
        """Saves model configuration to a JSON file."""
        config = {
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "encoder_config": self.encoder_config,
            "decoder_config": self.decoder_config,
            "vq_config": self.vq_config,
            "latent_dim": self.latent_dim,
            "encoder_downsampling": self.encoder_downsampling,
            "pad_factor_h": self.pad_factor_h,
            "pad_factor_w": self.pad_factor_w,
        }
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    @classmethod
    def from_pretrained(cls, repo_id_or_path, **kwargs):
        """Loads a pretrained model from Hugging Face Hub or local path."""
        from huggingface_hub import hf_hub_download
        import tempfile

        is_local = os.path.isdir(repo_id_or_path)

        if is_local:
            config_path = os.path.join(repo_id_or_path, "config.json")
            model_path = os.path.join(repo_id_or_path, "pytorch_model.bin")
            if not os.path.exists(config_path) or not os.path.exists(model_path):
                 raise FileNotFoundError(f"Model files not found in {repo_id_or_path}")
        else:
             # Assume Hugging Face repo ID
             try:
                 config_path = hf_hub_download(repo_id=repo_id_or_path, filename="config.json", **kwargs)
                 model_path = hf_hub_download(repo_id=repo_id_or_path, filename="pytorch_model.bin", **kwargs)
             except Exception as e:
                 raise IOError(f"Could not download model from {repo_id_or_path}: {e}")

        model = cls.from_config(config_path)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def save_pretrained(self, save_directory):
         """Saves model weights and config to a directory."""
         os.makedirs(save_directory, exist_ok=True)
         # Save config
         self.save_config(os.path.join(save_directory, "config.json"))
         # Save state dict
         torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
         print(f"Model saved to {save_directory}")
