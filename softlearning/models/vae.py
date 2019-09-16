import tensorflow as tf
import numpy as np

from tensorflow.keras import regularizers
tfk = tf.keras
tfkl = tf.keras.layers

"""
Training methods
"""


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
    # Calculate the KL divergence (difference between log of unit
    # normal prior and posterior)
    logpz = log_normal_pdf(z, 0., 0.)  # Prior PDF
    logqz_x = log_normal_pdf(z, mean, logvar)  # Posterior
    reconstruction_loss = logpx_z
    kl_divergence = logpz - logqz_x
    loss = reconstruction_loss + beta * kl_divergence
    return -tf.reduce_mean(loss)


def compute_elbo_loss_split(model, x, beta=1.0):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit, labels=model.preprocess(x))
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])    
    logpz = log_normal_pdf(z, 0., 0.)  # Prior PDF
    logqz_x = log_normal_pdf(z, mean, logvar)  # Posterior
    reconstruction_loss = logpx_z
    kl_divergence = logpz - logqz_x
    loss = reconstruction_loss + beta * kl_divergence
    return (
        -tf.reduce_mean(reconstruction_loss),
        -tf.reduce_mean(beta * kl_divergence)
    )


@tf.function
def compute_apply_gradients(model, x, optimizer, beta=1.0):
    with tf.GradientTape() as tape:
        loss = compute_elbo_loss(model, x, beta=beta)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


"""
VAE model definition
"""


class VAE(tfk.Model):
    def __init__(
            self,
            image_shape,
            latent_dim=16,
            kernel_regularizer=regularizers.l2(l=5e-4)):
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
                             kernel_regularizer=None,
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
                trainable=trainable,
                kernel_regularizer=kernel_regularizer
            ),
            tfkl.Conv2D(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                activation=tfkl.LeakyReLU(),
                trainable=trainable,
                kernel_regularizer=kernel_regularizer
            ),
            tfkl.Conv2D(
                filters=32,
                kernel_size=3,
                strides=(1, 1),
                activation=tfkl.LeakyReLU(),
                trainable=trainable,
                kernel_regularizer=kernel_regularizer
            ),
            tfkl.Flatten(),
            tfkl.Dense(
                latent_dim + latent_dim,
                trainable=trainable,
                kernel_regularizer=kernel_regularizer
            )
        ], name=name)

    def create_decoder_model(self,
                             latent_dim=None,
                             trainable=True,
                             kernel_regularizer=None,
                             name='decoder'):
        if latent_dim is None:
            latent_dim = self.latent_dim
        return tfk.Sequential([
            tfkl.InputLayer(input_shape=(latent_dim,)),
            # This layer expands the dimensionality a lot.
            tfkl.Dense(
                units=4*4*32,
                activation=tfkl.LeakyReLU(),
                trainable=trainable,
                kernel_regularizer=kernel_regularizer
            ),
            tfkl.Reshape(target_shape=(4, 4, 32)),
            tfkl.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation=tfkl.LeakyReLU(),
                trainable=trainable,
                kernel_regularizer=kernel_regularizer
            ),
            tfkl.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation=tfkl.LeakyReLU(),
                trainable=trainable,
                kernel_regularizer=kernel_regularizer
            ),
            tfkl.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation=tfkl.LeakyReLU(),
                trainable=trainable,
                kernel_regularizer=kernel_regularizer
            ),
            tfkl.Conv2DTranspose(
                filters=3,
                kernel_size=3,
                strides=(1, 1),
                padding="SAME",
                trainable=trainable,
                kernel_regularizer=kernel_regularizer
            )
        ], name=name)

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

    def reparameterize_split(self, mean_logvar_concat):
        mean, logvar = tf.split(
            mean_logvar_concat, num_or_size_splits=2, axis=1)
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

    def get_encoder_decoder(self, trainable=True, name='vae_encoder_decoder'):
        import ipdb; ipdb.set_trace()
        encoder = self.create_encoder_model(
            self.image_shape, trainable=trainable)
        encoder.set_weights(self.encoder.get_weights())
        decoder = self.create_decoder_model(
            self.latent_dim, trainable=trainable)
        decoder.set_weights(self.decoder.get_weights())

        def reparameterize_split(mean_logvar_concat):
            mean, logvar = tf.split(
                mean_logvar_concat, num_or_size_splits=2, axis=1)
            # eps = tf.random.normal(shape=mean.shape)
            # return tf.reshape(eps * tf.exp(logvar * .5) + mean, mean.shape)
            return mean

        encoder.add(tfkl.Lambda(reparameterize_split))
        encoder.add(decoder)
        model = tfk.Sequential([
            tfkl.InputLayer(input_shape=self.image_shape),
            encoder,
            tfkl.Lambda(self.reparameterize_split),
            decoder
        ], name=name)
        return model

    def get_encoder(self, trainable=True, name='encoder'):
        import ipdb; ipdb.set_trace()
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


"""
Training script
"""


def get_datasets(images,
                 batch_size=128,
                 shuffle=True,
                 validation_split_ratio=0.1):
    if shuffle:
        np.random.shuffle(images)
    split_index = int(validation_split_ratio * len(images))
    train_images = images[split_index:]
    test_images = images[:split_index]

    def train_generator():
        for image in train_images:
            yield image

    def test_generator():
        for image in test_images:
            yield image

    train_dataset = tf.data.Dataset.from_generator(
            train_generator, tf.uint8).batch(batch_size)
    test_dataset = tf.data.Dataset.from_generator(
            test_generator, tf.uint8).batch(batch_size)

    return train_dataset, test_dataset


if __name__ == '__main__':
    import argparse
    import time
    import os
    import skimage

    REGULARIZER_OPTIONS = {
        'l1': regularizers.l1,
        'l2': regularizers.l2,
        'l1_l2': regularizers.l1_l2,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-dim',
                        type=int,
                        help='Latent dimension of the VAE',
                        default=4)
    parser.add_argument('--image-shape',
                        type=lambda x: eval(x),
                        help='(width, height, channels) of the image',
                        default=(32, 32, 3))
    parser.add_argument('--save-path-name',
                        type=str,
                        help='Name to store the VAE weights under',
                        default='vae')
    parser.add_argument('--n-epochs',
                        type=int,
                        help='Number of epochs to train for',
                        default=250)
    parser.add_argument('--beta',
                        type=float,
                        help='Beta parameter for the VAE',
                        default=1.0)
    parser.add_argument('--n-examples-to-generate',
                        type=float,
                        help='Number of random reconstructions to generate',
                        default=4)
    parser.add_argument('--save-weights-frequency',
                        type=int,
                        help='Number of epochs between each save',
                        default=25)
    parser.add_argument('--data-directory',
                        type=str,
                        help='.pkl file to load the data from',
                        default='/root/nfs/kun1/users/justinvyu/data/fixed_data_with_states.pkl')
    parser.add_argument('--kernel-regularizer',
                        type=str,
                        help='Kernel regularizer to use for Conv2D and Dense layers',
                        choices=list(REGULARIZER_OPTIONS.keys()) + ['None'],
                        default='l2')
    parser.add_argument('--regularizer_lambda',
                        type=float,
                        help='Lambda to use with kernel regularizer',
                        default=5e-4)
    args = parser.parse_args()

    tf.enable_eager_execution()
    
    path_name = args.save_path_name
    n_epochs = args.n_epochs
    beta = args.beta
    image_shape = args.image_shape
    latent_dim = args.latent_dim
    n_examples_to_generate = args.n_examples_to_generate
    save_weights_frequency = args.save_weights_frequency
    kernel_regularizer_type = args.kernel_regularizer
    lambd = args.regularizer_lambda

    regularizer = REGULARIZER_OPTIONS[kernel_regularizer_type](l=lambd)

    cur_dir = os.getcwd()
    save_path = os.path.join(cur_dir, path_name)
    reconstruct_save_path = os.path.join(save_path, 'reconstructions')
    if not os.path.exists(reconstruct_save_path):
        os.makedirs(reconstruct_save_path)

    from softlearning.models.state_estimation import get_dumped_pkl_data
    images, _ = get_dumped_pkl_data(args.data_directory)

    train_dataset, test_dataset = get_datasets(images)

    optimizer = tf.keras.optimizers.Adam(1e-4)
    # Model creation
    vae = VAE(
        image_shape=image_shape,
        latent_dim=latent_dim,
        kernel_regularizer=regularizer
    )

    vae.encoder.summary()
    vae.decoder.summary()

    elbo_history = []
    recon_history, kl_history = [], []

    for epoch in range(1, n_epochs + 1):
        # Training loop
        start_time = time.time()
        for train_x in train_dataset:
            compute_apply_gradients(vae, train_x, optimizer, beta)
        end_time = time.time()

        if epoch % save_weights_frequency == 0:
            vae.encoder.save_weights(
                os.path.join(save_path, f'encoder_{latent_dim}_dim_{beta}_beta.h5'))
            vae.decoder.save_weights(
                os.path.join(save_path, f'decoder_{latent_dim}_dim_{beta}_beta.h5'))

        # Test on eval dataset
        _elbo = tf.keras.metrics.Mean()
        _recon_loss, _kl_loss = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
        for test_x in test_dataset:
            recon, kl = compute_elbo_loss_split(vae, test_x, beta=beta) 
            _elbo(recon + kl)
            _recon_loss(recon)
            _kl_loss(kl)
        elbo = -(_elbo.result())
        recon_loss = -(_recon_loss.result())
        kl_loss = -(_kl_loss.result())
        print(f'Epoch: {epoch}, Test set ELBO: {elbo}, Reconstruction Loss: {recon_loss}, KL loss: {kl_loss}\nTime elapsed for current epoch {end_time-start_time}')

        elbo_history.append(elbo)
        recon_history.append(recon_loss)
        kl_history.append(kl_loss)

        # Evaluate qualitatively on some reconstructions
        random_images = images[np.random.randint(images.shape[0],
                                                 size=n_examples_to_generate)]
        reconstructions = vae(random_images)
        for i, (r, orig) in enumerate(zip(reconstructions, random_images)):
            concat = np.concatenate([
                skimage.util.img_as_ubyte(r), orig], axis=1)
            img_path = os.path.join(reconstruct_save_path, f'epoch_{epoch}_{i}.png')
            skimage.io.imsave(img_path, concat)
