# Image SNAC (Inspired by SNAC Audio Codec) 🖼️

This project implements a **Multi-Scale Neural Image Codec** inspired by the [SNAC audio codec](https://github.com/hubertsiuzdak/snac). It compresses images into hierarchical discrete tokens using a Vector Quantized Variational Autoencoder (VQ-VAE) architecture with residual quantization stages, similar to SNAC, EnCodec, and SoundStream, but adapted for 2D image data.

## Overview

Like its audio counterpart, this model encodes an image into multiple streams of discrete tokens (codes), where different streams can represent features at different spatial resolutions (controlled by `vq_strides`). This hierarchical structure can be beneficial for generative modeling tasks.

The model aims to handle images of **variable sizes** by internally padding the input to be compatible with the network's downsampling factors and then cropping the output back to the original dimensions.

**Key Features:**

*   Encoder-Decoder architecture (VQ-VAE based).
*   Residual Vector Quantization (RVQ) with multiple codebooks.
*   Configurable spatial strides (`vq_strides`) for hierarchical token maps.
*   Designed to handle variable input image sizes (via padding/cropping).
*   Includes scripts for training, encoding, and decoding.

*(Note: This is an adaptation inspired by SNAC. Performance and behavior will differ from the original audio codec.)*

## Pretrained Models

*(Space for listing pretrained models once available)*

*   No models pretrained yet. Train your own using the provided script!

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd image_snac
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the package (optional but recommended):**
    ```bash
    pip install .
    ```

## Usage

### 1. Download Sample Data (Optional)

A script is provided to download a small set of images from Picsum Photos for initial testing:

```bash
python scripts/download_picsum.py --num_images 50 --size 256 --output_dir ./data/picsum
