
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
    output_size = 4
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
        output_activation=tf.keras.activations.tanh,
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(preprocessed)
   
    return PicklableModel(inputs, estimator_outputs, name='state_estimator_preprocessor')

def get_seed_data(seed_path):
    checkpoint_paths = [
        os.path.join(path, 'replay_pool.pkl')
        for path in sorted(glob.iglob(os.path.join(seed_path, 'checkpoint_*')))
    ]

    training_images = []
    ground_truth_states = []
    for checkpoint_path in checkpoint_paths:

        i = 0
        print(checkpoint_path)
        with gzip.open(checkpoint_path, 'rb') as f:
            pool = pickle.load(f)
            obs = pool['observations']
            training_images.append(obs['pixels'])
            pos = obs['object_position'][:, :2]
            pos = normalize(pos, -0.1, 0.1, -1, 1) 
            num_samples = pos.shape[0] 
            ground_truth_state = np.concatenate([
                pos,
                obs['object_orientation_cos'][:, 2].reshape((num_samples, 1)),
                obs['object_orientation_sin'][:, 2].reshape((num_samples, 1)),
            ], axis=1)
            ground_truth_states.append(ground_truth_state)

    return np.concatenate(training_images), np.concatenate(ground_truth_states)

def get_training_data(exp_path, limit=None): 
    for exp in sorted(glob.iglob(os.path.join(exp_path, '*'))):
        training_images = None # np.array([])
        ground_truth_states = None # np.array([])

        if not os.path.isdir(exp):
            continue
        print(exp)
        checkpoint_paths = [
            os.path.join(path, 'replay_pool.pkl')
            for path in sorted(glob.iglob(os.path.join(exp, 'checkpoint_*')))
        ]

        training_images = []
        ground_truth_states = []
        for checkpoint_path in checkpoint_paths:

            i = 0
            print(checkpoint_path)
            with gzip.open(checkpoint_path, 'rb') as f:
                pool = pickle.load(f)
                obs = pool['observations']
                training_images.append(obs['pixels'])
                pos = obs['object_position'][:, :2]
                pos = normalize(pos, -0.1, 0.1, -1, 1) 
                num_samples = pos.shape[0] 
                ground_truth_state = np.concatenate([
                    pos,
                    obs['object_orientation_cos'][:, 2].reshape((num_samples, 1)),
                    obs['object_orientation_sin'][:, 2].reshape((num_samples, 1)),
                ], axis=1)
                ground_truth_states.append(ground_truth_state)

            i += 1
            if limit is not None and i == limit:
                break
            
    training_images = np.concatenate(training_images, axis=0) 
    ground_truth_states = np.concatenate(ground_truth_states, axis=0)
    return training_images, ground_truth_states

def normalize(data, olow, ohigh, nlow, nhigh):
    """
    olow    old low
    ohigh   old high
    nlow    new low
    nhigh   new hight
    """
    percent = (data - olow) / (ohigh - olow)
    return percent * (nhigh - nlow) + nlow

def train(model, obs_keys_to_estimate):
    training_pools_base_path = '/home/justinvyu/ray_results/gym/DClaw/TurnFreeValve3ResetFreeSwapGoal-v0/2019-08-07T14-57-41-state_gtr_2_goals_with_resets_regular_box_saving_pixels_fixed_env'
    training_pools_base_path = '/home/justinvyu/ray_results/gym/DClaw/TurnFreeValve3ResetFreeSwapGoal-v0/2019-08-07T14-57-41-state_gtr_2_goals_with_resets_regular_box_saving_pixels_fixed_env/id=612875d0-seed=9463_2019-08-07_14-57-42op75_8n7'
    if 'seed' in training_pools_base_path:
        pixels, states = get_seed_data(training_pools_base_path)
    else:
        pixels, states = get_training_data(training_pools_base_path)

    model.fit(
        x=pixels,
        y=states,
        batch_size=64,
        epochs=30,
        validation_split=0.2,
    )

    import ipdb; ipdb.set_trace()

    random_indices = np.random.choice(training_images.shape[0], size=50, replace=False)
    tests, labels = training_images[random_indices], ground_truth_states[random_indices]
    preds = model.predict(tests)
    import imageio
    for n, test_img in enumerate(tests):
        imageio.imwrite(f'/tmp/test_obs/test{n}.jpg', test_img)

    print("Model error norm mean: ", np.mean(np.linalg.norm(labels - preds, axis=1)))
    model.save_weights('./state_estimator_model_single_seed.h5')

if __name__ == '__main__':
    image_shape = (64, 64, 3)

    obs_keys = ('object_position',
                'object_orientation_cos',
                'object_orientation_sin')
    model = state_estimator_model(
        domain='DClaw',
        task='TurnFreeValve3ResetFreeSwapGoal-v0',
        obs_keys_to_estimate=obs_keys, 
        input_shape=image_shape)
    
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='mean_squared_error')

    load_weights = True
    if load_weights:
        training_pools_base_path = '/home/justinvyu/ray_results/gym/DClaw/TurnFreeValve3ResetFreeSwapGoal-v0/2019-08-07T14-57-41-state_gtr_2_goals_with_resets_regular_box_saving_pixels_fixed_env'

        model.load_weights('./state_estimator_model_single_seed.h5')
        images, labels = get_training_data(training_pools_base_path, limit=1)
        tests = images[:50]
        preds = model.predict(tests)
        import imageio
        for n, test_img in enumerate(tests):
            imageio.imwrite(f'/tmp/test_obs/test{n}.jpg', test_img)
        import ipdb; ipdb.set_trace()
    else:
        train(model, obs_keys)
