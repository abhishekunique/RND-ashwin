from copy import deepcopy
from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update
import os

DEFAULT_KEY = '__DEFAULT_KEY__'

M = 256
N = 2

REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

"""
Policy params
"""

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, ) * N,
        'squash': True,
        'observation_keys': None,
        'observation_preprocessors_params': {}
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
})

MAX_PATH_LENGTH_PER_DOMAIN = {
    DEFAULT_KEY: 100,
    'Point2D': 100,
    'DClaw': 100,
}

NUM_EPOCHS_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Point2D': {
            DEFAULT_KEY: 100,
            'Maze-v0': 200,
            'BoxWall-v1': 200,
        },
        'Pusher2D': {
            DEFAULT_KEY: 300,
        },
        'DClaw': {
            DEFAULT_KEY: 500,
        }
    },
}


"""
Algorithm params
"""

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_n_episodes': 3,
        'eval_deterministic': True,
        'save_training_video_frequency': 5,
        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
        'eval_render_kwargs': {
            'width': 480,
            'height': 480,
            'mode': 'rgb_array',
        },
    },
}

ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
            'n_epochs': 200,
        }
    },
    'DDL': {
        'type': 'DDL',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
            'n_epochs': 200,

            'train_distance_fn_every_n_steps': tune.grid_search([16, 64]),

            'ext_reward_coeff': tune.grid_search([0.25, 0.5]),
            'normalize_ext_reward_gamma': tune.grid_search([1]),
            'use_env_intrinsic_reward': tune.grid_search([True]),
            'ddl_symmetric': tune.grid_search([False]),
            # 'ddl_clip_length': tune.grid_search([None, 20, 50]),
            'ddl_train_steps': tune.grid_search([2, 4, 10]),
            'ddl_batch_size': 256,

            #'rnd_int_rew_coeff': tune.grid_search([None, 1]),
        },
        'rnd_params': {
            'convnet_params': {
                'conv_filters': (16, 32, 64),
                'conv_kernel_sizes': (3, 3, 3),
                'conv_strides': (2, 2, 2),
                'normalization_type': None,
            },
            'fc_params': {
                'hidden_layer_sizes': (256, 256),
                'output_size': 512,
            },
        }
    },
    'DynamicsAwareEmbeddingDDL': {
        'type': 'DynamicsAwareEmbeddingDDL',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
            'n_epochs': 200,

            'train_distance_fn_every_n_steps': tune.grid_search([16, 64]),
            'ddl_batch_size': 256,

            # 'ext_reward_coeff': tune.grid_search([0.5, 1]),
            # 'normalize_ext_reward_gamma': tune.grid_search([0.99, 1]),
            # 'use_env_intrinsic_reward': tune.grid_search([True]),
            # 'rnd_int_rew_coeff': 0,
        },
    }
}

DEFAULT_NUM_EPOCHS = 200
NUM_CHECKPOINTS = 10

"""
Distance Estimator params
"""

DISTANCE_FN_PARAMS_BASE = {
    'type': 'feedforward_distance_fn',
    'kwargs': {
        'hidden_layer_sizes': (M, ) * N,
        'observation_keys': None,
        #'classifier_params': {
        #    'max_distance': tune.grid_search([20, 40, 100]),
        #    'bins': tune.grid_search([10, 20]),
        #},
        # 'embedding_dim': 2,
    }
}

DISTANCE_FN_KWARGS_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Point2D': {
            **{
                key: {'observation_keys': ('state_observation', )}
                for key in (
                    'Fixed-v0',
                    'SingleWall-v0',
                    'Maze-v0',
                    'BoxWall-v1',
                )
            },
            **{
                key: {'observation_keys': ('onehot_observation', )}
                for key in (
                    # 'Fixed-v0',
                    # 'SingleWall-v0',
                    # 'Maze-v0',
                    # 'BoxWall-v1',
                )
            },
        },
        'Pusher2D': {
            **{
                key: {'observation_keys': ('object_pos', )}
                # key: {'observation_keys': ('gripper_qpos', 'object_pos')}
                for key in (
                    'Simple-v0',
                )
            },
        },
        'DClaw': {
            # **{
            #     key: {'observation_keys': ('object_pos', )}
            #     for key in (
            #         'Simple-v0',
            #     )
            # },
        }
    }
}

ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_STATE = {
    'gym': {
        'Point2D': {
            # === Point Mass ===
            'Fixed-v0': {
                'action_scale': tune.grid_search([0.5]),
                'images_are_rgb': True,
                # 'init_pos_range': ((-2, -2), (-2, -2)), # Fixed reset
                'init_pos_range': None,             # Random reset
                'target_pos_range': ((2, 2), (2, 2)), # Set the goal to (x, y) = (2, 2)
                # 'target_pos_range': ((0, 0), (0, 0)), # Set the goal to (x, y) = (0, 0)
                'render_onscreen': False,

                # 'n_bins': 10,
                # 'show_discrete_grid': True,

                # 'observation_keys': ('onehot_observation', ),
                'observation_keys': ('state_observation', ),
            },
            'SingleWall-v0': {
                # 'boundary_distance': tune.grid_search([4, 8]),
                'action_scale': tune.grid_search([0.5]),
                'images_are_rgb': True,
                'init_pos_range': None,   # Random reset
                'target_pos_range': ((0, 3), (0, 3)), # Set the goal to (x, y) = (2, 2)
                'render_onscreen': False,
                'observation_keys': ('state_observation', ),
            },
            'BoxWall-v1': {
                'action_scale': tune.grid_search([0.5]),
                'images_are_rgb': True,
                'init_pos_range': ((-3, -3), (-3, -3)),
                # 'init_pos_range': None,   # Random reset
                # 'target_pos_range': ((3.5, 3.5), (3.5, 3.5)),
                'target_pos_range': ((3, 3), (3, 3)),
                'render_onscreen': False,
                'observation_keys': ('state_observation', ),
            },
            'Maze-v0': {
                'action_scale': tune.grid_search([0.5]),
                'images_are_rgb': True,

                'reward_type': 'none',
                'use_count_reward': tune.grid_search([True]),
                # 'show_discrete_grid': False,
                # 'n_bins': 50,

                # === EASY ===
                # 'wall_shape': 'easy-maze',
                # 'init_pos_range': ((-2.5, -2.5), (-2.5, -2.5)),
                # 'target_pos_range': ((2.5, -2.5), (2.5, -2.5)),
                # === MEDIUM ===
                'wall_shape': 'medium-maze',
                'init_pos_range': ((-3, -3), (-3, -3)),
                'target_pos_range': ((3, 3), (3, 3)),
                # === HARD ===
                # 'wall_shape': 'hard-maze',
                # 'init_pos_range': ((-3, -3), (-3, -3)),
                # 'target_pos_range': ((-0.5, 1.25), (-0.5, 1.25)),

                'render_onscreen': False,
                'observation_keys': ('state_observation', ),
            },
        },
        'Pusher2D': {
            'Simple-v0': {
                'init_qpos_range': ((0, 0, 0), (0, 0, 0)),
                'init_object_pos_range': ((1, 0), (1, 0)),
                'target_pos_range': ((2, 2), (2, 2)),
                'reset_gripper': True,
                'reset_object': True,
                'observation_keys': (
                    'gripper_qpos',
                    'gripper_qvel',
                    'object_pos',
                    # 'target_pos'
                ),
            },
        },
        'DClaw': {
            'LiftDDFixed-v0': {
                'init_qpos_range': tune.grid_search([
                    ((0, 0, 0.041, 1.017, 0, 0), (0, 0, 0.041, 1.017, 0, 0))
                ]),
                'target_qpos_range': [
                    (0, 0, 0.045, 0, 0, 0)
                ],
                'reward_keys_and_weights': {'sparse_position_reward': 1},
                'observation_keys': (
                    'object_position',
                    'object_quaternion',
                    'claw_qpos',
                    'last_action'
                ),
                # Camera settings for video
                'camera_settings': {
                    'distance': 0.35,
                    'elevation': -15,
                    'lookat': (0, 0, 0.05),
                },
            },
        },
    },
}


ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_VISION = {
    'gym': {
        'DClaw': {},
    },
}

"""
Helper methods for retrieving universe/domain/task specific params.
"""


def get_policy_params(universe, domain, task):
    policy_params = GAUSSIAN_POLICY_PARAMS_BASE.copy()
    return policy_params


def get_max_path_length(universe, domain, task):
    max_path_length = MAX_PATH_LENGTH_PER_DOMAIN.get(domain) or \
        MAX_PATH_LENGTH_PER_DOMAIN[DEFAULT_KEY]
    return max_path_length


def get_num_epochs(universe, domain, task):
    level_result = NUM_EPOCHS_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result
        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]
    return level_result


def get_environment_params(universe, domain, task, from_vision):
    if from_vision:
        params = ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_VISION
    else:
        params = ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_STATE

    environment_params = (
        params.get(universe, {}).get(domain, {}).get(task, {}))
    return environment_params


def get_distance_fn_params(universe, domain, task):
    distance_fn_params = DISTANCE_FN_PARAMS_BASE.copy()
    distance_fn_params['kwargs'].update(
        DISTANCE_FN_KWARGS_UNIVERSE_DOMAIN_TASK.get(
            universe, {}).get(domain, {}).get(task, {}))
    return distance_fn_params


def get_checkpoint_frequency(spec):
    config = spec.get('config', spec)
    checkpoint_frequency = (
        config
        ['algorithm_params']
        ['kwargs']
        ['n_epochs']
    ) // NUM_CHECKPOINTS

    return checkpoint_frequency


def is_image_env(universe, domain, task, variant_spec):
    return ('image' in task.lower()
            or 'image' in domain.lower()
            or 'pixel_wrapper_kwargs' in (
            variant_spec['environment_params']['training']['kwargs']))


"""
Preprocessor params
"""
STATE_PREPROCESSOR_PARAMS = {
    'ReplicationPreprocessor': {
        'type': 'ReplicationPreprocessor',
        'kwargs': {
            'n': 0,
            'scale_factor': 1,
        }
    },
    'RandomNNPreprocessor': {
        'type': 'RandomNNPreprocessor',
        'kwargs': {
            'hidden_layer_sizes': (32, 32),
            'activation': 'linear',
            'output_activation': 'linear',
        }
    },
    'RandomMatrixPreprocessor': {
        'type': 'RandomMatrixPreprocessor',
        'kwargs': {
            'output_size_scale_factor': 1,
            'coefficient_range': (-1., 1.),
        }
    },
    'None': None,
}


PIXELS_PREPROCESSOR_PARAMS = {
    'RAEPreprocessor': {
        'type': 'RAEPreprocessor',
        'kwargs': {
            'trainable': True,
            'image_shape': (32, 32, 3),
            'latent_dim': 32,
        },
        'shared': True,
    },
    'ConvnetPreprocessor': tune.grid_search([
        {
            'type': 'ConvnetPreprocessor',
            'kwargs': {
                'conv_filters': (64, ) * 3,
                'conv_kernel_sizes': (3, ) * 3,
                'conv_strides': (2, ) * 3,
                'normalization_type': normalization_type,
                'downsampling_type': 'conv',
                'output_kwargs': {
                    'type': 'flatten',
                }
            },
        }
        for normalization_type in (None, )
    ]),
}


"""
Configuring variant specs
"""


def get_variant_spec_base(universe, domain, task, task_eval,
                          policy, algorithm, from_vision):
    algorithm_params = ALGORITHM_PARAMS_BASE
    algorithm_params = deep_update(
            algorithm_params,
            ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
        )

    variant_spec = {
        'git_sha': get_git_rev(),
        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': get_environment_params(universe, domain, task, from_vision),
            },
            'evaluation': {
                'domain': domain,
                'task': task_eval,
                'universe': universe,
                'kwargs': (
                    tune.sample_from(lambda spec: (
                        spec.get('config', spec)
                        ['environment_params']
                        ['training']
                        .get('kwargs')
                    ))
                    if task == task_eval
                    else get_environment_params(universe, domain, task_eval, from_vision)),
            },
        },
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'exploration_policy_params': {
            'type': 'UniformPolicy',
            'kwargs': {
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                ))
            },
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                )),
                'observation_preprocessors_params': {}
            }
        },
        'distance_fn_params': get_distance_fn_params(universe, domain, task),
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                # 'max_size': int(5e5),
                'max_size': tune.grid_search([int(5e4)]),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': get_max_path_length(universe, domain, task),
                'min_pool_size': 50,
                'batch_size': 256, # tune.grid_search([128, 256]),
                'store_last_n_paths': 20,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': False,
            'checkpoint_frequency': tune.sample_from(get_checkpoint_frequency),
            'checkpoint_replay_pool': False,
        },
    }

    # Filter out parts of the state relating to the object when training from pixels
    env_kwargs = variant_spec['environment_params']['training']['kwargs']
    if from_vision and "device_path" not in env_kwargs.keys():
        env_obs_keys = env_kwargs.get('observation_keys', tuple())

        non_image_obs_keys = tuple(key for key in env_obs_keys if key != 'pixels')
        variant_spec['replay_pool_params']['kwargs']['obs_save_keys'] = non_image_obs_keys

        non_object_obs_keys = tuple(key for key in env_obs_keys if 'object' not in key)
        variant_spec['policy_params']['kwargs']['observation_keys'] = variant_spec[
            'exploration_policy_params']['kwargs']['observation_keys'] = variant_spec[
            'Q_params']['kwargs']['observation_keys'] = variant_spec[
            'distance_fn_params']['kwargs']['observation_keys'] = non_object_obs_keys

    return variant_spec

def get_variant_spec(args):
    universe, domain = args.universe, args.domain
    task, task_eval, algorithm = (
        args.task, args.task_evaluation, args.algorithm)
    from_vision = args.vision

    variant_spec = get_variant_spec_base(
        universe, domain, task, task_eval, args.policy, algorithm, from_vision)

    variant_spec['algorithm_params']['kwargs']['n_epochs'] = (
        get_num_epochs(universe, domain, task))

    preprocessor_type = args.preprocessor_type

    if is_image_env(universe, domain, task, variant_spec):
        assert preprocessor_type in PIXELS_PREPROCESSOR_PARAMS
        preprocessor_params = PIXELS_PREPROCESSOR_PARAMS[preprocessor_type]

        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (M, ) * N
        variant_spec['policy_params']['kwargs'][
            'observation_preprocessors_params'] = {
                'pixels': deepcopy(preprocessor_params)
            }

        variant_spec['Q_params']['kwargs']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['Q_params']['kwargs'][
            'observation_preprocessors_params'] = (
                tune.sample_from(lambda spec: (deepcopy(
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    ['observation_preprocessors_params']
                )))
            )

        variant_spec['distance_fn_params']['kwargs']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['distance_fn_params']['kwargs'][
            'observation_preprocessors_params'] = (
                tune.sample_from(lambda spec: (deepcopy(
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    ['observation_preprocessors_params']
                )))
            )

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
