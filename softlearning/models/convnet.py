import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

from softlearning.utils.keras import PicklableSequential
from softlearning.models.normalization import (
    LayerNormalization,
    GroupNormalization,
    InstanceNormalization)
from softlearning.utils.tensorflow import nest


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

POOLING_TYPES = {
    'avg_pool': layers.AvgPool2D,
    'max_pool': layers.MaxPool2D,
}

OUTPUT_TYPES = (
    'spatial_softmax',
    'dense',
    'flatten'
)
DEFAULT_OUTPUT_KWARGS = {'type': 'flatten'}


def convnet_model(
        conv_filters=(64, 64, 64),
        conv_kernel_sizes=(3, 3, 3),
        conv_strides=(2, 2, 2),
        padding="SAME",
        normalization_type=None,
        normalization_kwargs={},
        downsampling_type='conv',
        activation=layers.LeakyReLU,
        name="convnet",
        output_kwargs=None,
        *args,
        **kwargs):
    normalization_layer = {
        'batch': layers.BatchNormalization,
        'layer': LayerNormalization,
        'group': GroupNormalization,
        'instance': InstanceNormalization,
        None: None,
    }[normalization_type]

    def conv_block(conv_filter, conv_kernel_size, conv_stride):
        block_parts = [
            layers.Conv2D(
                filters=conv_filter,
                kernel_size=conv_kernel_size,
                strides=(conv_stride if downsampling_type == 'conv' else 1),
                padding=padding,
                activation='linear',
                *args,
                **kwargs),
        ]

        if normalization_layer is not None:
            block_parts += [normalization_layer(**normalization_kwargs)]

        block_parts += [(layers.Activation(activation)
                         if isinstance(activation, str)
                         else activation())]

        # if downsampling_type == 'pool' and conv_stride > 1:
        if downsampling_type in POOLING_TYPES:
            block_parts += [
                POOLING_TYPES[downsampling_type](
                    pool_size=conv_stride, strides=conv_stride
                )
            ]

        block = tfk.Sequential(block_parts, name='conv_block')
        return block

    def preprocess(x):
        """Cast to float, normalize, and concatenate images along last axis."""
        x = nest.map_structure(
            lambda image: tf.image.convert_image_dtype(image, tf.float32), x)
        x = nest.flatten(x)
        x = tf.concat(x, axis=-1)
        x = (tf.image.convert_image_dtype(x, tf.float32) - 0.5) * 2.0
        return x

    output_kwargs = output_kwargs or DEFAULT_OUTPUT_KWARGS
    output_type = output_kwargs.get('type', DEFAULT_OUTPUT_KWARGS['type'])
    if output_type == 'spatial_softmax':
        def spatial_softmax(x):
            # Create learnable temperature parameter `alpha`
            alpha = tf.Variable(1., dtype=tf.float32, name='softmax_alpha')
            width, height, channels = x.shape[1:]
            x_flattened = tf.reshape(
                x, [-1, width * height, channels])
            softmax_attention = tf.math.softmax(x_flattened / alpha, axis=1)
            # TODO: Fix this; redundant, since I'm going to reflatten it later
            softmax_attention = tf.reshape(
                softmax_attention, [-1, width, height, channels])
            return softmax_attention

        def calculate_expectation(distributions):
            width, height, channels = distributions.shape[1:]

            # Create matrices where all xs/ys are the same value acros
            # the row/col. These will be multiplied by the softmax distr
            # to get the 2D expectation.
            pos_x, pos_y = tf.meshgrid(
                tf.linspace(-1., 1., num=width),
                tf.linspace(-1., 1., num=height),
                indexing='ij'
            )
            # Reshape to a column vector to satisfy multiply broadcast.
            pos_x, pos_y = (
                tf.reshape(pos_x, [-1, 1]),
                tf.reshape(pos_y, [-1, 1])
            )

            distributions = tf.reshape(
                distributions, [-1, width * height, channels])

            expected_x = tf.math.reduce_sum(
                pos_x * distributions, axis=[1], keepdims=True)
            expected_y = tf.math.reduce_sum(
                pos_y * distributions, axis=[1], keepdims=True)
            expected_xy = tf.concat([expected_x, expected_y], axis=1)
            feature_keypoints = tf.reshape(expected_xy, [-1, 2 * channels])
            return feature_keypoints

        # def spatial_softmax(x):
        #     # Create learnable temperature parameter `alpha`
        #     alpha = tf.Variable(1., dtype=tf.float32, name='softmax_alpha')
        #     width, height, channels = x.shape[1:]
        #     # softmax_attention = tf.math.softmax(x / alpha)
        #     # Create matrices where all xs/ys are the same value acros
        #     # the row/col. These will be multiplied by the softmax distr
        #     # to get the 2D expectation.
        #     pos_x, pos_y = tf.meshgrid(
        #         tf.linspace(-1., 1., num=width),
        #         tf.linspace(-1., 1., num=height),
        #         indexing='ij'
        #     )
        #     # Reshape to a column vector to satisfy multiply broadcast.
        #     pos_x, pos_y = (
        #         tf.reshape(pos_x, [-1, 1]),
        #         tf.reshape(pos_y, [-1, 1])
        #     )
        #     # Vectorize the feature maps, split by channels still
        #     # softmax_attention = tf.reshape(
        #     #     softmax_attention, [-1, width * height, channels])
        #     x_flattened = tf.reshape(
        #         x, [-1, width * height, channels])
        #     softmax_attention = tf.math.softmax(x_flattened / alpha, axis=1)

        #     expected_x = tf.math.reduce_sum(
        #         pos_x * softmax_attention, axis=[1], keepdims=True)
        #     expected_y = tf.math.reduce_sum(
        #         pos_y * softmax_attention, axis=[1], keepdims=True)
        #     expected_xy = tf.concat([expected_x, expected_y], axis=1)
        #     feature_keypoints = tf.reshape(expected_xy, [-1, 2 * channels])

        #     return feature_keypoints

        # output_layer = tfkl.Lambda(spatial_softmax)

        output_layer = tfk.Sequential([
            tfkl.Lambda(spatial_softmax),
            tfkl.Lambda(calculate_expectation)
        ])
    elif output_type == 'dense':
        # TODO: Implement this with `feedforward` network
        pass
    else:
        output_layer = tfkl.Flatten()

    model = PicklableSequential((
        tfkl.Lambda(preprocess),
        *[
            conv_block(conv_filter, conv_kernel_size, conv_stride)
            for (conv_filter, conv_kernel_size, conv_stride) in
            zip(conv_filters, conv_kernel_sizes, conv_strides)
        ],
        output_layer,
    ), name=name)
    return model
