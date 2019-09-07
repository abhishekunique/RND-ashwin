from copy import deepcopy
from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update
from softlearning.misc.generate_goal_examples import (
    DOOR_TASKS, PUSH_TASKS, PICK_TASKS)
from softlearning.misc.get_multigoal_example_pools import (
    get_example_pools_from_variant)
import dsuite

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

"""
Policy params
"""

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
        'observation_keys': None, # can specify some keys to look at
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

DEFAULT_MAX_PATH_LENGTH = 100
MAX_PATH_LENGTH_PER_DOMAIN = {
    'DClaw': 100, # 50
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
        'eval_render_kwargs': {},
        'eval_n_episodes': 3,
        'eval_deterministic': True,
        'save_training_video_frequency': 5,
        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
        'normalize_ext_reward_gamma': 0.99,
        'rnd_int_rew_coeff': tune.sample_from([1]),
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
}

# TODO(Avi): Most of the algorithm params for classifier-style methods
# are shared. Rewrite this part to reuse the params
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
    'SACClassifier': {
        'type': 'SACClassifier',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 10000,
            'classifier_optim_name': 'adam',
            'reward_type': 'logits',
            'n_epochs': 200,
            'mixup_alpha': 1.0,
        }
    },
    'RAQ': {
        'type': 'RAQ',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 10,
            'classifier_optim_name': 'adam',
            'reward_type': 'logits',
            'active_query_frequency': 1,
            'n_epochs': 200,
            'mixup_alpha': 1.0,
        }
    },
    'VICE': {
        'type': 'VICE',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 5, # tune.grid_search([1, 5]),
            'classifier_optim_name': 'adam',
            'n_epochs': 200,
            'mixup_alpha': 1.0,
            'save_training_video_frequency': 5,
        }
    },
    'VICEGANTwoGoal': {
        'type': 'VICEGANTwoGoal',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 5, # tune.grid_search([2, 5]),
            'classifier_optim_name': 'adam',
            'n_epochs': 500,
            'mixup_alpha': 1.0,
            'save_training_video_frequency': 5,
        }
    },
    'VICEGANMultiGoal': {
        'type': 'VICEGANMultiGoal',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 5,
            'classifier_optim_name': 'adam',
            'n_epochs': 500,
            'mixup_alpha': 1.0,
            'save_training_video_frequency': 5,
        }
    },
    'VICEGAN': {
        'type': 'VICEGAN',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 10,
            'classifier_optim_name': 'adam',
            'n_epochs': 200,
            'mixup_alpha': 1.0,
        }
    },
    'VICERAQ': {
        'type': 'VICERAQ',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 10,
            'classifier_optim_name': 'adam',
            'active_query_frequency': 1,
            'n_epochs': 200,
            'mixup_alpha': 1.0,
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'td_target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                }[spec.get('config', spec)['domain']],
            ))
        }
    }
}

DEFAULT_NUM_EPOCHS = 200
NUM_CHECKPOINTS = 10

"""
Environment params
"""

GOALS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'DClaw': {
            'TurnFreeValve3MultiGoalResetFree-v0': {
                '2': (
                    (0.01, 0.01, 0, 0, 0, np.pi),
                    (0.01, 0.01, 0, 0, 0, np.pi),
                ),
                '4': (
                    (0.01, 0.01, 0, 0, 0, 0),
                    (0.01, -0.01, 0, 0, 0, np.pi / 2),
                    (-0.01, -0.01, 0, 0, 0, np.pi),
                    (-0.01, 0.01, 0, 0, 0, -np.pi / 2),
                ),
            },
            # TODO: Remove this redundancy
            'TurnFreeValve3MultiGoal-v0': {
                '2': (
                    (0.01, 0.01, 0, 0, 0, np.pi),
                    (0.01, 0.01, 0, 0, 0, np.pi),
                ),
                '4': (
                    (0.01, 0.01, 0, 0, 0, 0),
                    (0.01, -0.01, 0, 0, 0, np.pi / 2),
                    (-0.01, -0.01, 0, 0, 0, np.pi),
                    (-0.01, 0.01, 0, 0, 0, -np.pi / 2),
                ),
            }
        }
    }
}

CLASSIFIER_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {

}

ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_STATE = {
    'gym': {
        'DClaw': {
            'TurnResetFree-v0': {
                'init_object_pos_range': (0., 0.),
                'target_pos_range': (-np.pi, np.pi),
                'reward_keys': ('object_to_target_angle_dist_cost', )
            },
            'TurnFreeValve3ResetFree-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 2,
                    'object_to_target_orientation_distance_reward': 1,
                },
                'reset_fingers': True,
                'swap_goal_upon_completion': False,
            },
            'TurnFreeValve3Fixed-v0': {
                'init_angle_range': (-np.pi, np.pi),
                'target_angle_range': (np.pi, np.pi),     # Sample in this range
                # 'target_angle_range': [np.pi, 0.],          # Sample from one of these 2 goals
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 2,
                    'object_to_target_orientation_distance_reward': 1,
                },
            },
            'TurnFreeValve3MultiGoalResetFree-v0': {
                # 'goals': ((0, 0, 0, 0, 0, np.pi), (0, 0, 0, 0, 0, 0)),
                'goals': (
                    (0.01, 0.01, 0, 0, 0, 0),
                    (0.01, -0.01, 0, 0, 0, np.pi / 2),
                    (-0.01, -0.01, 0, 0, 0, np.pi),
                    (-0.01, 0.01, 0, 0, 0, -np.pi / 2),
                ),
                'goal_completion_position_threshold': 0.04,
                'goal_completion_orientation_threshold': 0.15,
                'swap_goals_upon_completion': False,
            },
            'TurnFreeValve3MultiGoal-v0': {
                # 'goals': ((0, 0, 0, 0, 0, np.pi), (0, 0, 0, 0, 0, 0)),
                'goals': (
                    (0.01, 0.01, 0, 0, 0, 0),
                    (0.01, -0.01, 0, 0, 0, np.pi / 2),
                    (-0.01, -0.01, 0, 0, 0, np.pi),
                    (-0.01, 0.01, 0, 0, 0, -np.pi / 2),
                ),
                'swap_goals_upon_completion': False,
                'random_goal_sampling': True,
            },
            'TurnFreeValve3ResetFreeSwapGoal': {
                'init_angle_range': (0., 0.),
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_cost': 2,
                    'object_to_target_orientation_distance_cost': 1,
                },
            },
            'TurnFreeValve3ResetFreeSwapGoalEval': {
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_cost': 2,
                    'object_to_target_orientation_distance_cost': 1,
                },
            },
        }
    },
}

ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_VISION = {
    'gym': {
        'DClaw': {
            'TurnFixed-v0': {
                'init_object_pos_range': (0., 0.),
                'target_pos_range': (np.pi, np.pi),
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                    'normalize': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                        'camera_id': -1,
                    }
                },
                'camera_settings': {
                    'azimuth': 0.,
                    'distance': 0.35,
                    'elevation': -38.17570837642188,
                    'lookat': np.array([0.00046945, -0.00049496, 0.05389398]),
                },
                'observation_keys': ('pixels', 'claw_qpos', 'last_action'),
            },
            'TurnMultiGoalResetFree-v0': { # training environment
                'goals': (np.pi, 0.), # Two goal setting
                # 'goals': (2 * np.pi / 3, 4 * np.pi / 3, 0.), #np.arange(0, 2 * np.pi, np.pi / 3),
                # 'goals': np.arange(0, 2 * np.pi, np.pi / 2), # 4 goal setting
                'initial_goal_index': 0, # start with np.pi
                'swap_goals_upon_completion': False, # if false, will swap at every reset
                'use_concatenated_goal': False,
                'one_hot_goal_index': False, # True, use the scalar goal index
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                    'normalize': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                        'camera_id': -1,
                    }
                },
                'camera_settings': {
                    'azimuth': 0.,
                    'distance': 0.32,
                    'elevation': -45.18,
                    'lookat': np.array([0.00047, -0.0005, 0.060])
                },
                'observation_keys': ('pixels', 'claw_qpos', 'last_action', 'goal_index'),
            },
            'TurnMultiGoal-v0': { # eval environment
                'goals': (np.pi, 0.),
                # 'goals': np.arange(0, 2 * np.pi, np.pi / 2),
                'initial_goal_index': 0,
                'swap_goals_upon_completion': False,
                'use_concatenated_goal': False,
                'one_hot_goal_index': False, # True
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                    'normalize': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                        'camera_id': -1
                    }
                },
                'camera_settings': {
                    'azimuth': 0.,
                    'distance': 0.32,
                    'elevation': -45.18,
                    'lookat': np.array([0.00047, -0.0005, 0.060])
                },
                'observation_keys': ('pixels', 'claw_qpos', 'last_action', 'goal_index'),
            },
            'TurnFreeValve3ResetFree-v0': {
                'pixel_wrapper_kwargs': {
                   'pixels_only': False,
                   'normalize': False,
                   'render_kwargs': {
                       'width': 48,
                       'height': 48,
                       'camera_id': -1,
                   }
                },
                'camera_settings': {
                    'azimuth': 45,
                    'distance': 0.32,
                    'elevation': -55.88,
                    'lookat': np.array([0.00097442, 0.00063182, 0.03435371]),
                },
                # 'camera_settings': {
                #    'azimuth': 0.,
                #    'distance': 0.35,
                #    'elevation': -38.17570837642188,
                #    'lookat': np.array([0.00046945, -0.00049496, 0.05389398]),
                # },
                'init_angle_range': (0., 0.),
                'target_angle_range': (np.pi, np.pi),
                'swap_goal_upon_completion': False,
                'observation_keys': ('pixels', 'claw_qpos', 'last_action', 'object_position', 'object_orientation_sin', 'object_orientation_cos'),
            },
            'TurnFreeValve3Fixed-v0': {
               'pixel_wrapper_kwargs': {
                   'pixels_only': False,
                   'normalize': False,
                   'render_kwargs': {
                       'width': 48,
                       'height': 48,
                       'camera_id': -1,
                   }
               },
               'camera_settings': {
                   'azimuth': 0.,
                   'distance': 0.35,
                   'elevation': -38.17570837642188,
                   'lookat': np.array([ 0.00046945, -0.00049496,  0.05389398]),
               },
               'init_angle_range': (0., 0.),
               'target_angle_range': (np.pi, np.pi),
               'observation_keys': ('pixels', 'claw_qpos', 'last_action'),
            },
            'TurnFreeValve3MultiGoalResetFree-v0': {
                'goals': ((0, 0, 0, 0, 0, np.pi), (0, 0, 0, 0, 0, 0)),
                'goal_completion_position_threshold': 0.05,
                'goal_completion_orientation_threshold': 0.15,
                'camera_settings': {
                    'azimuth': 45.,
                    'distance': 0.32,
                    'elevation': -55.88,
                    'lookat': np.array([0.00097442, 0.00063182, 0.03435371])
                },
                # 'camera_settings': {
                #     'azimuth': 30.,
                #     'distance': 0.35,
                #     'elevation': -38.18,
                #     'lookat': np.array([0.00047, -0.0005, 0.054])
                # },
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                    'normalize': False,
                    'render_kwargs': {
                        'width': 48,
                        'height': 48,
                        'camera_id': -1
                    },
                },
                'swap_goals_upon_completion': False,
                'observation_keys': ('pixels', 'claw_qpos', 'last_action', 'goal_index', 'object_position', 'object_orientation_sin', 'object_orientation_cos'),
            },
            'TurnFreeValve3MultiGoal-v0': {
                'goals': ((0, 0, 0, 0, 0, np.pi), (0, 0, 0, 0, 0, 0)),
                'camera_settings': {
                    'azimuth': 45.,
                    'distance': 0.32,
                    'elevation': -55.88,
                    'lookat': np.array([0.00097442, 0.00063182, 0.03435371])
                },
                # 'camera_settings': {
                #     'azimuth': 30.,
                #     'distance': 0.35,
                #     'elevation': -38.18,
                #     'lookat': np.array([0.00047, -0.0005, 0.054])
                # },
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                    'normalize': False,
                    'render_kwargs': {
                        'width': 48,
                        'height': 48,
                        'camera_id': -1
                    },
                },
                'swap_goals_upon_completion': False,
                'random_goal_sampling': True,
                'observation_keys': ('pixels', 'claw_qpos', 'last_action', 'goal_index'),
            },
        }
    },
}

"""
Helper methods for retrieving universe/domain/task specific params.
"""

def get_policy_params(universe, domain, task):
    policy_params = GAUSSIAN_POLICY_PARAMS_BASE.copy()
    return policy_params

def get_max_path_length(universe, domain, task):
    max_path_length = MAX_PATH_LENGTH_PER_DOMAIN.get(
        task, DEFAULT_MAX_PATH_LENGTH)
    return max_path_length

def get_environment_params(universe, domain, task, from_pixels):
    if from_pixels:
        params = ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_VISION
    else:
        params = ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_STATE

    environment_params = (
        params.get(universe, {}).get(domain, {}).get(task, {}))
    return environment_params

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
Configuring variant specs
"""

def get_variant_spec_base(universe, domain, task, task_eval,
                          policy, algorithm, from_pixels):
    algorithm_params = ALGORITHM_PARAMS_BASE
    algorithm_params = deep_update(
            algorithm_params,
            ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
        )

    variant_spec = {
        'git_sha': get_git_rev(),
        # 'num_goals': 2, # TODO: Separate classifier_rl with multigoal
        'num_goals': 4,
        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': get_environment_params(universe, domain, task, from_pixels),
            },
            'evaluation': {
                'domain': domain,
                'task': task_eval,
                'universe': universe,
                'kwargs': get_environment_params(universe, domain, task_eval, from_pixels),
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
                )), # None means everything, pass in all keys but the goal_index
                'observation_preprocessors_params': {}
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': int(2e5), #int(1e6)
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': get_max_path_length(universe, domain, task),
                'min_pool_size': get_max_path_length(universe, domain, task),
                'batch_size': 256,
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

    env_kwargs = variant_spec['environment_params']['training']['kwargs']
    if "pixel_wrapper_kwargs" in env_kwargs.keys() and \
       "device_path" not in env_kwargs.keys():
        env_obs_keys = env_kwargs['observation_keys']

        non_image_obs_keys = tuple(key for key in env_obs_keys if key != 'pixels')
        variant_spec['replay_pool_params']['kwargs']['obs_save_keys'] = non_image_obs_keys

        non_object_obs_keys = tuple(key for key in env_obs_keys if 'object' not in key)
        variant_spec['policy_params']['kwargs']['observation_keys'] = variant_spec[
            'exploration_policy_params']['kwargs']['observation_keys'] = variant_spec[
                'Q_params']['kwargs']['observation_keys'] = non_object_obs_keys

    return variant_spec


def get_variant_spec_classifier(universe,
                                domain,
                                task,
                                task_eval,
                                policy,
                                algorithm,
                                n_goal_examples,
                                from_pixels,
                                *args,
                                **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, task_eval, policy, algorithm, from_pixels, *args, **kwargs)

    classifier_layer_size = L = 256
    variant_spec['reward_classifier_params'] = {
        'type': 'feedforward_classifier',
        'kwargs': {
            'hidden_layer_sizes': (L, L),
            }
        }

    # TODO: Abstract this
    variant_spec['reward_classifier_params']['kwargs']['observation_keys'] = ('object_position', 'object_orientation_cos', 'object_orientation_sin', 'goal_index')

    variant_spec['data_params'] = {
        'n_goal_examples': n_goal_examples,
        'n_goal_examples_validation_max': 100,
    }

    if algorithm in ['RAQ', 'VICERAQ']:

        if task in DOOR_TASKS:
            is_goal_key = 'angle_success'
        elif task in PUSH_TASKS:
            is_goal_key = 'puck_success'
        elif task in PICK_TASKS:
            is_goal_key = 'obj_success'
        else:
            raise NotImplementedError('Success metric not defined for task')

        variant_spec.update({
            'sampler_params': {
                'type': 'ActiveSampler',
                'kwargs': {
                    'is_goal_key': is_goal_key,
                    'max_path_length': get_max_path_length(universe, domain, task),
                    'min_pool_size': get_max_path_length(universe, domain, task),
                    'batch_size': 256,
                }
            },
            'replay_pool_params': {
                'type': 'ActiveReplayPool',
                'kwargs': {
                    'max_size': 1e6,
                }
            },
        })

    return variant_spec

CLASSIFIER_ALGS = ('SACClassifier', 'RAQ', 'VICE', 'VICEGAN', 'VICERAQ', 'VICEGANTwoGoal', 'VICEGANMultiGoal')

def get_variant_spec(args):
    universe, domain = args.universe, args.domain
    task, task_eval, algorithm, n_epochs = (
        args.task, args.task_evaluation, args.algorithm, args.n_epochs)

    from_pixels = args.from_pixels

    if args.algorithm in CLASSIFIER_ALGS:
        variant_spec = get_variant_spec_classifier(
            universe, domain, task, task_eval, args.policy, algorithm,
            args.n_goal_examples, from_pixels)
    else:
        variant_spec = get_variant_spec_base(
            universe, domain, task, task_eval, args.policy, algorithm, from_pixels)

    if args.algorithm in ('RAQ', 'VICERAQ'):
        active_query_frequency = args.active_query_frequency
        variant_spec['algorithm_params']['kwargs'][
            'active_query_frequency'] = active_query_frequency

    variant_spec['algorithm_params']['kwargs']['n_epochs'] = n_epochs

    if is_image_env(universe, domain, task, variant_spec):
        preprocessor_params = {
            'type': 'ConvnetPreprocessor',
            'kwargs': {
                'conv_filters': (64, ) * 3,
                'conv_kernel_sizes': (3, ) * 3,
                'conv_strides': (2, ) * 3,
                'normalization_type': None, # 'layer',
                'downsampling_type': 'conv',
            },
        }

        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (M, M)
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
        if args.algorithm in CLASSIFIER_ALGS:
            (variant_spec
             ['reward_classifier_params']
             ['kwargs']
             ['observation_preprocessors_params']) = (
                tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    ['observation_preprocessors_params']
                )))

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)
   
    return variant_spec
