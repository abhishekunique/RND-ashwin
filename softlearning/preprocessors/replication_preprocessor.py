import tensorflow as tf
from tensorflow.keras.backend import repeat_elements
from tensorflow.python.keras.engine import training_utils

from softlearning.utils.keras import PicklableSequential
from softlearning.utils.tensorflow import nest

tfkl = tf.keras.layers

def replication_preprocessor(
        n=1,
        scale_factor=1,
        name='replication_preprocessor'):
    # def cast_and_concat(x):
    #     x = nest.map_structure(training_utils.cast_if_floating_dtype, x)
    #     x = nest.flatten(x)
    #     x = tf.concat(x, axis=-1)
    #     return x

    def replicate_and_scale(x):
        x = tf.tile(x, [1, n])
        x = scale_factor * x
        return x

    return tfkl.Lambda(replicate_and_scale)
