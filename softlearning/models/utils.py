from collections import OrderedDict
from copy import deepcopy

import tensorflow as tf

from softlearning.utils.tensorflow import nest
from softlearning.preprocessors.utils import get_preprocessor_from_params
from .vice_models import create_feedforward_reward_classifier_function


def get_reward_classifier_from_variant(variant, env, *args, **kwargs):
    reward_classifier_params = deepcopy(variant['reward_classifier_params'])
    reward_classifier_type = deepcopy(reward_classifier_params['type'])
    assert reward_classifier_type == 'feedforward_classifier', (
        reward_classifier_type)
    reward_classifier_kwargs = deepcopy(reward_classifier_params['kwargs'])

    observation_preprocessors_params = reward_classifier_kwargs.pop(
        'observation_preprocessors_params', {}).copy()
    observation_keys = reward_classifier_kwargs.pop(
        'observation_keys', None) or env.observation_keys

    observation_shapes = OrderedDict((
        (key, value) for key, value in env.observation_shape.items()
        if key in observation_keys
    ))
    action_shape = env.action_shape
    input_shapes = {
        'observations': observation_shapes,
        'actions': action_shape,
    }

    observation_preprocessors = OrderedDict()
    for name, observation_shape in observation_shapes.items():
        preprocessor_params = observation_preprocessors_params.get(name, None)
        if not preprocessor_params:
            observation_preprocessors[name] = None
            continue
        observation_preprocessors[name] = get_preprocessor_from_params(
            env, preprocessor_params)

    action_preprocessor = None
    preprocessors = {
        'observations': observation_preprocessors,
        'actions': action_preprocessor,
    }

    reward_classifier = create_feedforward_reward_classifier_function(
        input_shapes=input_shapes,
        observation_keys=observation_keys,
        *args,
        preprocessors=preprocessors,
        **reward_classifier_kwargs,
        **kwargs)

    return reward_classifier


def get_inputs_for_nested_shapes(input_shapes, name=None):
    if isinstance(input_shapes, dict):
        return type(input_shapes)([
            (name, get_inputs_for_nested_shapes(value, name))
            for name, value in input_shapes.items()
        ])
    elif isinstance(input_shapes, (tuple, list)):
        if all(isinstance(x, int) for x in input_shapes):
            return tf.keras.layers.Input(shape=input_shapes, name=name)
        else:
            return type(input_shapes)((
                get_inputs_for_nested_shapes(input_shape, name=None)
                for input_shape in input_shapes
            ))
    elif isinstance(input_shapes, tf.TensorShape):
        return tf.keras.layers.Input(shape=input_shapes, name=name)

    raise NotImplementedError(input_shapes)


def flatten_input_structure(inputs):
    inputs_flat = nest.flatten(inputs)
    return inputs_flat


def create_input(name, input_shape):
    input_ = tf.keras.layers.Input(
        shape=input_shape,
        name=name,
        dtype=(tf.uint8 # Image observation
               if len(input_shape) == 3 and input_shape[-1] in (1, 3)
               else tf.float32) # Non-image
    )
    return input_


def create_inputs(input_shapes):
    """Creates `tf.keras.layers.Input`s based on input shapes.

    Args:
        input_shapes: (possibly nested) list/array/dict structure of
        inputs shapes.

    Returns:
        inputs: nested structure, of same shape as input_shapes, containing
        `tf.keras.layers.Input`s.

    TODO(hartikainen): Need to figure out a better way for handling the dtypes.
    """
    inputs = nest.map_structure_with_paths(create_input, input_shapes)
    inputs_flat = flatten_input_structure(inputs)

    return inputs_flat
