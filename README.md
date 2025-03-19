# Generative Adversarial Networks (GANs) with PyTorch

## Project Overview
This project explores the implementation of **Generative Adversarial Networks (GANs)** using **PyTorch**. We train models to generate realistic images from the MNIST dataset of handwritten digits. Specifically, we implement three types of GANs:

1. **Vanilla GAN**: The basic form of a GAN using fully connected layers.
2. **Least Squares GAN (LSGAN)**: A modified loss function for more stable training and improved image quality.
3. **Deeply Convolutional GAN (DCGAN)**: A GAN that uses convolutional layers to capture spatial hierarchies in images, leading to sharper and more realistic results.

## Files and Structure

- `gan_pytorch.py`: Contains helper functions, loss functions, and the network architectures for the generator and discriminator.
- `gan-checks.npz`: Includes validation data for testing the correctness of implemented functions.
- `gan_outputs_pytorch.png`: Shows sample outputs from the trained models.
- `images/`: Directory storing images generated during training.

## Models
### 1. Vanilla GAN
- The generator transforms a noise vector into a 28x28 image, and the discriminator distinguishes between real and generated images.
- **Loss function**: Binary Cross-Entropy Loss.
- Results: Early training results in noisy images, but after ~3000 iterations, the generator starts producing recognizable digit shapes.

### 2. Least Squares GAN (LSGAN)
- A modification of the GAN with least squares loss for the generator and discriminator, which helps to stabilize training.
- **Loss function**: Least Squares Loss.
- Results: LSGAN tends to converge faster, with sharper and clearer digit images than the Vanilla GAN.

### 3. Deeply Convolutional GAN (DCGAN)
- Utilizes convolutional layers to improve the quality of generated images by learning spatial hierarchies.
- **Loss function**: Binary Cross-Entropy Loss (for comparison).
- Results: DCGAN produces the best results with clearly defined, sharp digits.

## Requirements

- **Python 3.7+**
- **PyTorch 1.7.0+**
- **Torchvision 0.8.1+**
- **NumPy**
- **Matplotlib**

To install all dependencies:
```bash
pip install -r requirements.txt
