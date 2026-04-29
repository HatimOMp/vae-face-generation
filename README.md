# 🧠 VAE Face Generation

A deep generative model that creates realistic human faces from random noise, built with a **Variational Autoencoder (VAE)** trained on the CelebA dataset.

![Sample Generated Faces](sample_generations.png)

## 🎯 Project Overview

This project implements a VAE from scratch using TensorFlow/Keras. The model learns a compressed latent representation of human faces and can generate new, never-seen-before faces by sampling from that latent space.

**Key concepts implemented:**
- Convolutional encoder/decoder architecture
- Reparameterization trick for backpropagation through stochastic nodes
- KL divergence + reconstruction loss (ELBO)
- Latent space interpolation and exploration
- Interactive Streamlit deployment

## 🏗️ Architecture
Input Image (64x64x3)
↓
Encoder (CNN)
Conv2D x3 + Dense
↓
Latent Space (z=128)
z_mean + z_log_var
Reparameterization
↓
Decoder (CNN)
Dense + ConvTranspose2D x3
↓
Generated Image (64x64x3)

## 📊 Results

| Metric | Value |
|--------|-------|
| Dataset | CelebA (20,000 images) |
| Image size | 64 × 64 px |
| Latent dimensions | 128 |
| Epochs | 30 |
| Final loss | ~2162 |

The model successfully learns to generate diverse faces with varying:
- Hair color (blonde, brunette, dark)
- Skin tone
- Gender
- Facial expression

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/vae-face-generation
cd vae-face-generation
```

**2. Create a virtual environment**
```bash
conda create -n vae-env python=3.10
conda activate vae-env
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the Streamlit app**
```bash
streamlit run app.py
```

## 🗂️ Project Structure
vae-face-generation/
│
├── vae.py                        # VAE model architecture
├── train.py                      # Training script (run on Google Colab)
├── app.py                        # Streamlit web interface
├── requirements.txt              # Dependencies
└── sample_generations.png        # Generated faces after training

## 🛠️ Tech Stack

- **TensorFlow / Keras** — model architecture and training
- **Python** — data pipeline and preprocessing
- **Streamlit** — interactive web interface
- **Google Colab** — GPU training (T4)
- **CelebA Dataset** — 200,000+ celebrity face images

## 📚 Key Concepts

**Variational Autoencoder:** Unlike a standard autoencoder, a VAE learns a *distribution* over the latent space rather than a fixed encoding. This allows meaningful sampling and interpolation.

**Reparameterization trick:** To allow gradients to flow through the stochastic sampling step, we express `z = μ + ε × σ` where `ε ~ N(0,1)`. This makes the sampling differentiable.

**ELBO Loss:** The training objective combines reconstruction loss (how well the decoder rebuilds the input) and KL divergence (how close the learned distribution is to a standard normal).

## 👤 Author

**Hatim Omari** — [LinkedIn](https://www.linkedin.com/in/hatim-omari/) · [GitHub](https://github.com/HatimOMp)