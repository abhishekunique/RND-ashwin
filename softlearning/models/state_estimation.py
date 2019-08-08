
from collections import OrderedDict
from softlearning.models.convnet import convnet_model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from softlearning.models.feedforward import feedforward_model
from softlearning.utils.keras import PicklableModel, PicklableSequential
from softlearning.preprocessors.utils import get_preprocessor_from_params
import numpy as np
from softlearning.environments.adapters.gym_adapter import GymAdapter
import gzip
import pickle
import glob
import os

def state_estimator_model(domain, task, obs_keys_to_estimate, input_shape):
    """
    Need to pass in the obs_keys that the model will estimate
    """
    env = GymAdapter(domain=domain, task=task)
    output_sizes = OrderedDict(
        (key, value)
        for key, value in env.observation_shape.items()
        if key in obs_keys_to_estimate
    )
    output_size = np.sum([size[0].value for size in output_sizes.values()])

    num_layers = 4
    normalization_type = None
    convnet_kwargs = {
        'conv_filters': (64, ) * num_layers,
        'conv_kernel_sizes': (3, ) * num_layers,
        'conv_strides': (2, ) * num_layers,
        'normalization_type': normalization_type,
    }
    preprocessor = convnet_model(name='convnet_preprocessor', **convnet_kwargs)

    
    inputs = Input(shape=input_shape)
    preprocessed = preprocessor(inputs)
    estimator_outputs = feedforward_model(
        hidden_layer_sizes=(256, 256),
        output_size=output_size,
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(preprocessed)
   
    # model = PicklableSequential((
    # ), name='state_estimator_preprocessor')

    return PicklableModel(inputs, estimator_outputs, name='state_estimator_preprocessor')

def train(model, obs_keys_to_estimate):
    training_pools_base_path = '/home/justinvyu/ray_results/gym/DClaw/TurnFreeValve3ResetFreeSwapGoal-v0/2019-08-05T15-41-14-state_gtr_2_goals_with_resets_regular_box_saving_pixels/'
    
    for exp in sorted(glob.iglob(os.path.join(training_pools_base_path, '*'))):
        training_images = None # np.array([])
        ground_truth_states = None # np.array([])

        if not os.path.isdir(exp):
            continue
        print(exp)
        checkpoint_paths = [
            os.path.join(path, 'replay_pool.pkl')
            for path in sorted(glob.iglob(os.path.join(exp, 'checkpoint_*')))
        ]
        for checkpoint_path in checkpoint_paths:
            with gzip.open(checkpoint_path, 'rb') as f:
                pool = pickle.load(f)
                if training_images is None:
                    training_images = pool['observations']['pixels']
                else:
                    training_images = np.concatenate((training_images, pool['observations']['pixels']), axis=0)
                ground_truth_state = np.concatenate(
                    [pool['observations'][key] for key in obs_keys_to_estimate],
                    axis=1
                )
                if ground_truth_states is None:
                    ground_truth_states = ground_truth_state
                else:
                    ground_truth_states = np.concatenate((ground_truth_states, ground_truth_state), axis=0)
                
            print(checkpoint_path, training_images.shape, ground_truth_states.shape)
        
        model.fit(
            x=training_images,
            y=ground_truth_states,
            batch_size=512,
            epochs=20,
            validation_split=0.1,
        )


    # with gzip.open(path, 'rb') as f:
    #     data = pickle.load(f)

    # obs = OrderedDict(data['observations'])
    # pixels = obs['pixels']

    # ground_truth_state = np.concatenate(
    #     [obs[key] for key in obs_keys_to_estimate],
    #     axis=1
    # ) 
    example, label = np.array([training_images[-1]]), ground_truth_states[-1]
    pred = model.predict(example)
    import imageio
    imageio.imwrite('/tmp/test_obs/test.jpg', example[0])
    print("ACTUAL STATE: ", label)
    print("PREDICTION: ", pred)
    model.save_weights('./state_estimator_model_all_replay_pools.h5')

if __name__ == '__main__':
    image_shape = (32, 32, 3)

    obs_keys = ('object_position', 'object_orientation_cos', 'object_orientation_sin')
    model = state_estimator_model('DClaw', 'TurnFreeValve3ResetFreeSwapGoal-v0', obs_keys_to_estimate=obs_keys, input_shape=image_shape)
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')

    train(model, obs_keys)

