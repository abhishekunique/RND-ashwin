from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim=64):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ])

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=8*8*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(8, 8, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation=tf.keras.layers.LeakyReLU()),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=3, strides=(1, 1), padding="SAME"),
                ]
            )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
        return probs

      return logits

    def __call__(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_reconstruct = self.decode(z, apply_sigmoid=True)
        return x_reconstruct

    def load_weights(self, encoder_weights, decoder_weights):
        self.encoder.load_weights(encoder_weights)
        self.decoder.load_weights(decoder_weights)

