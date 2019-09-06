import tensorflow as tf
import numpy as np

tfk = tf.keras
tfkl = tf.keras.layers


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def compute_elbo_loss(model, x, beta=1.0):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    # Cross entropy reconstruction loss assumes that the pixels
    # are all independent Bernoulli r.v.s
    # Need to preprocess the label, so the output will be normalized.
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit, labels=model.preprocess(x))
    # Sum across all pixels (row/col) + channels
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    # Calculate the KL divergence (difference between log of unit normal prior and posterior)
    logpz = log_normal_pdf(z, 0., 0.) # Prior PDF
    logqz_x = log_normal_pdf(z, mean, logvar) # Posterior
    reconstruction_loss = logpx_z
    kl_divergence = logpz - logqz_x
    loss = reconstruction_loss + beta * kl_divergence
    return -tf.reduce_mean(loss)

class VAE(tfk.Model):
    def __init__(self, image_shape, latent_dim=16):
        super().__init__()
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.encoder = self.create_encoder_model()
        self.decoder = self.create_decoder_model()

    def preprocess(self, x):
        # Turn integers into floats normalized between [0, 1]
        x = tf.image.convert_image_dtype(x, tf.float32)
        return x

    def create_encoder_model(self,
                             image_shape=None,
                             latent_dim=None,
                             trainable=True,
                             name='encoder'):
        if image_shape is None:
            image_shape = self.image_shape
        if latent_dim is None:
            latent_dim = self.latent_dim
        return tfk.Sequential([
            tfkl.InputLayer(input_shape=image_shape),
            tfkl.Lambda(self.preprocess),
            tfkl.Conv2D(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                activation=tfkl.LeakyReLU(),
                trainable=trainable
            ),
            tfkl.Conv2D(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                activation=tfkl.LeakyReLU(),
                trainable=trainable
            ),
            tfkl.Conv2D(
                filters=32,
                kernel_size=3,
                strides=(1, 1),
                activation=tfkl.LeakyReLU(),
                trainable=trainable
            ),
            tfkl.Flatten(),
            tfkl.Dense(latent_dim + latent_dim, trainable=trainable)
        ], name=name)
    
    def create_decoder_model(self, latent_dim=None):
        if latent_dim is None:
            latent_dim = self.latent_dim
        return tfk.Sequential([
            tfkl.InputLayer(input_shape=(latent_dim,)),
            # This layer expands the dimensionality a lot.
            tfkl.Dense(units=4*4*32, activation=tfkl.LeakyReLU()),
            tfkl.Reshape(target_shape=(4, 4, 32)),
            tfkl.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation=tfkl.LeakyReLU()),
            tfkl.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation=tfkl.LeakyReLU()),
            tfkl.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation=tfkl.LeakyReLU()),
            tfkl.Conv2DTranspose(
                filters=3,
                kernel_size=3,
                strides=(1, 1),
                padding="SAME")
        ], name='decoder')
        
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(
            self.encoder(x), num_or_size_splits=2, axis=1)
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
    
    def get_encoder(self, trainable=True, name='encoder'):
        encoder = self.create_encoder_model(
            self.image_shape, trainable=trainable, name=name)
        # Copy weights over to this new model
        encoder.set_weights(self.encoder.get_weights())
        # Only return the mean (no need to sample an epsilon)
        def get_encoded_mean(mean_logvar_concat):
            mean, logvar = tf.split(
                mean_logvar_concat, num_or_size_splits=2, axis=1)
            return mean
        encoder.add(tfkl.Lambda(get_encoded_mean, name='encoded_mean'))
        encoder.summary()
        return encoder
        
        
