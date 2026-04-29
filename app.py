import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import os

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="VAE Face Generator",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 VAE Face Generator")
st.markdown("Generate human faces using a **Variational Autoencoder** trained on CelebA (200,000+ images).")

# ── Model definition (must match training) ───────────────────────────
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(latent_dim=128):
    inputs = tf.keras.Input(shape=(64, 64, 3))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

def build_decoder(latent_dim=128):
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 128, activation="relu")(inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="decoder")

# ── Load model ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    encoder = build_encoder()
    decoder = build_decoder()
    dummy = tf.zeros((1, 64, 64, 3))
    encoder(dummy)
    decoder(tf.zeros((1, 128)))
    if os.path.exists("encoder_weights.weights.h5") and os.path.exists("decoder_weights.weights.h5"):
        encoder.load_weights("encoder_weights.weights.h5")
        decoder.load_weights("decoder_weights.weights.h5")
        return encoder, decoder, True
    return encoder, decoder, False

encoder, decoder, weights_loaded = load_model()

if not weights_loaded:
    st.warning("⚠️ Weight files not found. Showing untrained generations.")

def generate_faces(n=1):
    z = tf.random.normal(shape=(n, 128))
    return decoder(z).numpy()

# ── Generation ───────────────────────────────────────────────────────
st.subheader("🎲 Generate Random Faces")
n_faces = st.slider("Number of faces to generate", 1, 9, 4)

if st.button("Generate", type="primary"):
    with st.spinner("Generating..."):
        imgs = generate_faces(n=n_faces)
        cols = st.columns(min(n_faces, 3))
        for i, img in enumerate(imgs):
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            cols[i % 3].image(pil_img, use_column_width=True)

# ── Latent space exploration ─────────────────────────────────────────
st.subheader("🔀 Explore the Latent Space")
st.markdown("Adjust sliders to manually explore the latent space.")

if "latent_vector" not in st.session_state:
    st.session_state.latent_vector = np.zeros(128)

cols = st.columns(5)
for i in range(10):
    with cols[i % 5]:
        st.session_state.latent_vector[i] = st.slider(
            f"z[{i}]", -3.0, 3.0,
            float(st.session_state.latent_vector[i]), 0.1
        )

z = tf.constant(st.session_state.latent_vector[np.newaxis], dtype=tf.float32)
generated = decoder(z).numpy()[0]
pil_img = Image.fromarray((generated * 255).astype(np.uint8))
st.image(pil_img, caption="Generated from custom latent vector", width=256)

# ── Sample output ────────────────────────────────────────────────────
st.subheader("📸 Training Sample Output")
if os.path.exists("sample_generations.png"):
    st.image("sample_generations.png", caption="Faces generated after 30 epochs of training")

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built with TensorFlow & Streamlit · [GitHub](https://github.com/HatimOm/vae-face-generation)")