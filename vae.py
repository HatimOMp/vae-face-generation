import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# ── Sampling layer ───────────────────────────────────────────────────
class Sampling(layers.Layer):
    """Reparameterization trick: z = mean + eps * exp(0.5 * log_var)"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ── Encoder ──────────────────────────────────────────────────────────
def build_encoder(latent_dim=128, input_shape=(64, 64, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")


# ── Decoder ──────────────────────────────────────────────────────────
def build_decoder(latent_dim=128):
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 128, activation="relu")(inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(3, 3, padding="same", activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="decoder")


# ── VAE model ────────────────────────────────────────────────────────
class VAE(tf.keras.Model):
    def __init__(self, latent_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = build_encoder(latent_dim)
        self.decoder = build_decoder(latent_dim)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Reconstruction loss
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {"loss": self.total_loss_tracker.result()}

    def generate(self, n=1):
        """Generate n faces from random latent vectors."""
        z = tf.random.normal(shape=(n, self.latent_dim))
        return self.decoder(z).numpy()

    def interpolate(self, img1, img2, steps=8):
        """Interpolate between two images in latent space."""
        z1, _, _ = self.encoder(img1[np.newaxis])
        z2, _, _ = self.encoder(img2[np.newaxis])
        alphas = np.linspace(0, 1, steps)
        imgs = [self.decoder((1 - a) * z1 + a * z2).numpy()[0] for a in alphas]
        return imgs