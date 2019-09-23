from copy import deepcopy
from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update
from softlearning.misc.generate_goal_examples import (
    DOOR_TASKS, PUSH_TASKS, PICK_TASKS)
from softlearning.misc.get_multigoal_example_pools import (
    get_example_pools_from_variant)
import dsuite
import os

DEFAULT_KEY = '__DEFAULT_KEY__'

# M = 256
# N = 2
M = 512
N = 2

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

MAX_PATH_LENGTH_PER_DOMAIN = {
    DEFAULT_KEY: 100,
    'DClaw': 100,
    # 'DClaw': tune.grid_search([50, 100, 150]), # 100, # 50
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
        # 'normalize_ext_reward_gamma': 0.99,
        # 'rnd_int_rew_coeff': tune.sample_from([0]),
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
            'n_classifier_train_steps': 2, # tune.grid_search([2, 5]),
            'classifier_optim_name': 'adam',
            'n_epochs': 200,
            'mixup_alpha': 1.0,
            'save_training_video_frequency': 0,
        }
    },
    'MultiVICEGAN': {
        'type': 'MultiVICEGAN',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto', #tune.sample_from([-3, -5, -7]),#'auto',
            'action_prior': 'uniform',
            'her_iters': tune.grid_search([0]),
            'rnd_int_rew_coeffs': tune.sample_from([[1, 1]]),
            'ext_reward_coeffs': [1, 0], # 0 corresponds to reset policy
            'normalize_ext_reward_gamma': 0.99,
            'share_pool': False,
            'n_classifier_train_steps': 5,
            'classifier_optim_name': 'adam',
            'n_epochs': 500,
            'mixup_alpha': 1.0,
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
        },
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
            'n_classifier_train_steps': tune.grid_search([5]), # 5,
            'classifier_optim_name': 'adam',
            'n_epochs': 500,
            'mixup_alpha': 1.0,
            'save_training_video_frequency': 5,

            # RND options inherited from SAC
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
        },

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
            'n_classifier_train_steps': tune.grid_search([5]), # ==== TUNE THIS ====
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

CLASSIFIER_PARAMS_BASE = {
    'type': 'feedforward_classifier',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'observation_keys': ('pixels', ),
    }

}
CLASSIFIER_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'DClaw': {
            **{
                key: {'observation_keys': ('pixels', 'goal_index')}
                for key in (
                    'TurnMultiGoalResetFree-v0',
                    'TurnFreeValve3ResetFreeSwapGoal-v0',
                )
            },
        }
    }
}

ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_STATE = {
    'gym': {
        'DClaw': {
            # FIXED SCREW
            'TurnMultiGoalResetFree-v0': {
                'goals': (
                    -np.pi / 2,
                    np.pi / 2
                ),
                'initial_goal_index': 0,
                'swap_goals_upon_completion': False,
                'use_concatenated_goal': False,
                'one_hot_goal_index': True,
                'reward_keys_and_weights': {
                    'object_to_target_angle_dist': 1,
                },
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'goal_index',
                    # 'one_hot_goal_index',
                    # 'object_angle_cos',
                    # 'object_angle_sin',
                ),
            },
            'TurnMultiGoal-v0': { # eval environment
                'goals': (
                    -np.pi / 2,
                    np.pi / 2
                ),
                'initial_goal_index': 0,
                'swap_goals_upon_completion': False,
                'use_concatenated_goal': False,
                'one_hot_goal_index': True,
                'reward_keys_and_weights': {
                    'object_to_target_angle_dist': 1,
                },
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'goal_index',
                    # 'one_hot_goal_index',
                    # 'object_angle_cos',
                    # 'object_angle_sin',
                ),
            },

            'TurnResetFree-v0': {
                'init_object_pos_range': (0., 0.),
                'target_pos_range': (-np.pi, np.pi),
                'reward_keys': ('object_to_target_angle_dist_cost', )
            },

            # FREE SCREW
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
                'goals': ((0, 0, 0, 0, 0, np.pi), (0, 0, 0, 0, 0, 0)),
                # 'goals': (
                #     (0.01, 0.01, 0, 0, 0, 0),
                #     (0.01, -0.01, 0, 0, 0, np.pi / 2),
                #     (-0.01, -0.01, 0, 0, 0, np.pi),
                #     (-0.01, 0.01, 0, 0, 0, -np.pi / 2),
                # ),
                'goal_completion_position_threshold': 0.04,
                'goal_completion_orientation_threshold': 0.15,
                'swap_goals_upon_completion': False,
            },
            'TurnFreeValve3MultiGoal-v0': {
                'goals': ((0, 0, 0, 0, 0, np.pi), (0, 0, 0, 0, 0, 0)),
                # 'goals': (
                #     (0.01, 0.01, 0, 0, 0, 0),
                #     (0.01, -0.01, 0, 0, 0, np.pi / 2),
                #     (-0.01, -0.01, 0, 0, 0, np.pi),
                #     (-0.01, 0.01, 0, 0, 0, -np.pi / 2),
                # ),
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

            # LIFTING
            'LiftDDFixed-v0': {
                'reset_policy_checkpoint_path': None,
                'init_qpos_range': (
                    (0, 0, 0.041, 1.017, 0, 0),
                    (0, 0, 0.041, 1.017, 0, 0),
                    # (0, 0, 0.041, -np.pi, -np.pi, -np.pi),
                    # (0, 0, 0.041, np.pi, np.pi, np.pi),
                ),
                'target_qpos_range': [
                    (0, 0, 0.04, 0, 0, 0)
                ],
            }
        }
    },
}

BASE_VISION_KWARGS = {
    # 'pixel_wrapper_kwargs': {
    #     'pixels_only': False,
    #     'normalize': False,
    #     'render_kwargs': {
    #        'width': 32,
    #        'height': 32,
    #        'camera_id': -1,
    #     }
    # },
    # 'camera_settings': {
    #     'azimuth': 180,
    #     'distance': 0.35,
    #     'elevation': -55,
    #     'lookat': np.array([0, 0, 0.03]),
    # },
    'pixel_wrapper_kwargs': {
        'pixels_only': False,
        'normalize': False,
        'render_kwargs': {
           'width': 64,
           'height': 64,
           'camera_id': -1,
        }
    },
    'camera_settings': {
        'azimuth': 180,
        'distance': 0.38,
        'elevation': -36,
        'lookat': np.array([0.04, 0.008, 0.026]),
    },
}
FIXED_SCREW_VISION_KWARGS = {
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
        'azimuth': 180,
        'distance': 0.3,
        'elevation': -50,
        'lookat': np.array([0.02, 0.004, 0.09]),
    },
}

ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_VISION = {
    'gym': {
        'DClaw': {
            'TurnFixed-v0': {
                **FIXED_SCREW_VISION_KWARGS,
                # 'reward_keys_and_weights': {
                #     'object_to_target_angle_distance_reward': 1,
                # },
                'target_pos_range': (np.pi, np.pi),
                'init_pos_range': (-np.pi, np.pi),
                # 'camera_settings': {
                #     'azimuth': 0.,
                #     'distance': 0.35,
                #     'elevation': -38.17570837642188,
                #     'lookat': np.array([0.00046945, -0.00049496, 0.05389398]),
                # },
                # 'pixel_wrapper_kwargs': {
                #     'observation_key': 'pixels',
                #     'pixels_only': False,
                #     'render_kwargs': {
                #         'width': 32,
                #         'height': 32,
                #     },
                # },
                'observation_keys': (
                    'pixels',
                    'claw_qpos',
                    'last_action',
                    # == BELOW JUST FOR LOGGING ==
                    'object_angle_cos',
                    'object_angle_sin',
                ),
            },
            # === FIXED SCREW RESET FREE TASK BELOW ===
            'TurnResetFree-v0': {
                **FIXED_SCREW_VISION_KWARGS,
                'reward_keys_and_weights': {
                    # 'object_to_target_angle_distance_reward': 1,
                    'object_to_target_angle_distance_reward': 0, # Just to make sure 0 ext reward
                },
                'reset_fingers': True,
                'init_pos_range': (0, 0),
                'observation_keys': (
                    'claw_qpos',
                    'pixels',
                    'last_action',
                    # === BELOW JUST FOR LOGGING ===
                    'object_angle_cos',
                    'object_angle_sin',
                ),
            },
            # Random evaluation environment for free screw 
            'TurnFreeValve3Fixed-v0': {
                **BASE_VISION_KWARGS,
                'init_qpos_range': (
                    # (-0.075, -0.075, 0, 0, 0, -np.pi),
                    # (0.075, 0.075, 0, 0, 0, np.pi)
                    (-0.08, -0.08, 0, 0, 0, -np.pi),
                    (0.08, 0.08, 0, 0, 0, np.pi)
                ),
                'target_qpos_range': [
                    (0, 0, 0, 0, 0, -np.pi / 2),
                    (0, 0, 0, 0, 0, -np.pi / 2), # Second goal is arbitrary
                ],
                'observation_keys': (
                    'pixels',
                    'claw_qpos',
                    'last_action',
                    # === BELOW IS JUST FOR LOGGING ===
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                ),
            },
            'TurnFreeValve3ResetFree-v0': {
                **BASE_VISION_KWARGS,
                'init_qpos_range': [(0, 0, 0, 0, 0, 0)],
                # Below needs to be 2 for a MultiVICEGAN run, since the goals switch
                'target_qpos_range': [
                    (0, 0, 0, 0, 0, -np.pi / 2),
                    (0, 0, 0, 0, 0, -np.pi / 2), # Second goal is arbitrary
                ],
                'swap_goal_upon_completion': False,
                'observation_keys': (
                    'pixels',
                    'claw_qpos',
                    'last_action',
                    # === BELOW IS JUST FOR LOGGING ===
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                ),
            },
            'TurnFreeValve3ResetFreeSwapGoal-v0': {
                **BASE_VISION_KWARGS,
                'reset_fingers': True,
                'reset_frequency': 0,
                'goals': [
                    (0, 0, 0, 0, 0, np.pi / 2),
                    (0, 0, 0, 0, 0, -np.pi / 2),
                ],
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'target_xy_position',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                    'goal_index',
                    'pixels',
                    # === BELOW IS JUST FOR LOGGING ===
                    'object_xy_position',
                    'object_orientation_cos',
                    'object_orientation_sin',
                ),
            },
            'TurnFreeValve3ResetFreeSwapGoalEval-v0': {
                **BASE_VISION_KWARGS,
                'goals': [
                    (0, 0, 0, 0, 0, np.pi / 2),
                    (0, 0, 0, 0, 0, -np.pi / 2),
                ],
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'target_xy_position',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                    'goal_index',
                    'pixels',
                    # === BELOW IS JUST FOR LOGGING ===
                    'object_xy_position',
                    'object_orientation_cos',
                    'object_orientation_sin',
                ),
            },
            'LiftDDFixed-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 1,
                    'object_to_target_xy_position_distance_reward': 0,
                    'object_to_target_orientation_distance_reward': 0, #tune.sample_from([1, 5]), #5,
                },
                'target_qpos_range': [(0, 0, 0.05, 0, 0, 0)],
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                    },
                },
                'observation_keys': (
                    'claw_qpos',
                    'object_position',
                    'object_quaternion',
                    'last_action',
                    'target_position',
                    'target_quaternion',
                    'pixels',
                ),
                'camera_settings': {
                    'azimuth': 180,
                    'distance': 0.26,
                    'elevation': -40,
                    'lookat': (0, 0, 0.06),
                }
            },
            'LiftDDResetFree-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 1,
                    'object_to_target_xy_position_distance_reward': 0,
                    'object_to_target_orientation_distance_reward': 0, #tune.sample_from([1, 5]), #5,
                },
                # 'reset_policy_checkpoint_path': '/home/abhigupta/ray_results/gym/DClaw/LiftDDResetFree-v0/2019-08-12T22-28-02-random_translate/id=1efced72-seed=3335_2019-08-12_22-28-03bqyu82da/checkpoint_1500/',
                # 'target_qpos_range': (
                #      (-0.1, -0.1, 0.0, 0, 0, 0),
                #      (0.1, 0.1, 0.0, 0, 0, 0), # bgreen side up
                #  ),
                'target_qpos_range': [(0, 0, 0.05, 0, 0, 0)],
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                    },
                },
                'observation_keys': (
                    'claw_qpos',
                    'object_position',
                    'object_quaternion',
                    'last_action',
                    'target_position',
                    'target_quaternion',
                    'pixels',
                ),
                'camera_settings': {
                    'azimuth': 180,
                    'distance': 0.26,
                    'elevation': -40,
                    'lookat': (0, 0, 0.06),
                }
            },
            # Sliding Tasks
            'SlideBeadsFixed-v0': {
                'reward_keys_and_weights': {
                    'objects_to_targets_mean_distance_reward': 1,
                },
                'num_objects': 4,
                'init_qpos_range': [
                    (-0.0475, -0.0475, -0.0475, -0.0475),
                    (0.0475, 0.0475, 0.0475, 0.0475),
                ],
                'target_qpos_range': [
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                ],
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                    },
                },
                'observation_keys': (
                    'claw_qpos',
                    'objects_positions',
                    'last_action',
                    'objects_target_positions',
                    'pixels',
                ),
                'camera_settings': {
                    'azimuth': 90,
                    'lookat': (0,  0.04581637, -0.01614516),
                    'elevation': -45,
                    'distance': 0.37,
                },
                # 'camera_settings': {
                #     'azimuth': 23.234042553191497,
                #     'distance': 0.2403358053524018,
                #     'elevation': -29.68085106382978,
                #     'lookat': (-0.00390331,  0.01236683,  0.01093447),
                # }
            },
            'SlideBeadsResetFree-v0': {
                'reward_keys_and_weights': {
                    'objects_to_targets_mean_distance_reward': 1,
                },
                'init_qpos_range': [(0, 0)],
                'num_objects': 4,
                'target_qpos_range': [
                    (0, 0, 0, 0),
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                ],
                'cycle_goals': True,
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                    },
                },
                'observation_keys': (
                    'claw_qpos',
                    'objects_positions',
                    'last_action',
                    'objects_target_positions',
                    'pixels',
                ),
                # 'camera_settings': {
                #     'azimuth': 23.234042553191497,
                #     'distance': 0.2403358053524018,
                #     'elevation': -29.68085106382978,
                #     'lookat': (-0.00390331,  0.01236683,  0.01093447),
                # }
                'camera_settings': {
                    'azimuth': 90,
                    'lookat': (0,  0.04581637, -0.01614516),
                    'elevation': -45,
                    'distance': 0.37,
                },
            },
            'SlideBeadsResetFreeEval-v0': {
                'reward_keys_and_weights': {
                    'objects_to_targets_mean_distance_reward': 1,
                },
                'init_qpos_range': [(0, 0)],
                'num_objects': 4,
                'target_qpos_range': [
                    (0, 0, 0, 0),
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                ],
                # 'target_qpos_range': [
                #     (0, 0),
                #     (-0.0825, 0.0825),
                #     (0.0825, 0.0825),
                #     (-0.04, 0.04),
                #     (-0.0825, -0.0825),
                # ],
                'cycle_goals': True,
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                    },
                },
                'observation_keys': (
                    'claw_qpos',
                    'objects_positions',
                    'last_action',
                    'objects_target_positions',
                    'pixels',
                ),
                'camera_settings': {
                    'azimuth': 90,
                    'lookat': (0,  0.04581637, -0.01614516),
                    'elevation': -45,
                    'distance': 0.37,
                },
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
    max_path_length = MAX_PATH_LENGTH_PER_DOMAIN.get(domain) or \
        MAX_PATH_LENGTH_PER_DOMAIN[DEFAULT_KEY]
    return max_path_length


def get_environment_params(universe, domain, task, from_vision):
    if from_vision:
        params = ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_VISION
    else:
        params = ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_STATE

    environment_params = (
        params.get(universe, {}).get(domain, {}).get(task, {}))
    return environment_params


def get_classifier_params(universe, domain, task):
    classifier_params = CLASSIFIER_PARAMS_BASE.copy()
    classifier_params['kwargs'].update(
        CLASSIFIER_PARAMS_PER_UNIVERSE_DOMAIN_TASK.get(
            universe, {}).get(domain, {}).get(task, {}))
    return classifier_params


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


from softlearning.misc.utils import PROJECT_PATH, NFS_PATH
PIXELS_PREPROCESSOR_PARAMS = {
    'StateEstimatorPreprocessor': {
        'type': 'StateEstimatorPreprocessor',
        'kwargs': {
            'input_shape': (32, 32, 3),
            'num_hidden_units': 512,
            'num_hidden_layers': 2,
            'state_estimator_path': os.path.join(PROJECT_PATH,
                                                'softlearning',
                                                'models',
                                                'state_estimators',
                                                'state_estimator_from_vae_latents.h5'),
            # === INCLUDE A PRETRAINED VAE ===
            'preprocessor_params': {
                'type': 'VAEPreprocessor',
                'kwargs': {
                    'encoder_path': os.path.join(PROJECT_PATH,
                                                'softlearning',
                                                'models',
                                                'vae_16_dim_beta_3_invisible_claw_l2_reg',
                                                'encoder_16_dim_3.0_beta.h5'),
                    'decoder_path': os.path.join(PROJECT_PATH,
                                                'softlearning',
                                                'models',
                                                'vae_16_dim_beta_3_invisible_claw_l2_reg',
                                                'decoder_16_dim_3.0_beta.h5'),
                    'trainable': False,
                    'image_shape': (32, 32, 3),
                    'latent_dim': 16,
                    'include_decoder': False,
                }
            }
        }
    },
    'VAEPreprocessor': {
        'type': 'VAEPreprocessor',
        'kwargs': {
            # 'image_shape': (32, 32, 3),
            # 'latent_dim': 16,
            # 'encoder_path': '/nfs/kun1/users/justinvyu/pretrained_models/vae_16_dim_beta_3_invisible_claw_l2_reg/encoder_16_dim_3.0_beta.h5',
            # 'decoder_path': '/nfs/kun1/users/justinvyu/pretrained_models/vae_16_dim_beta_3_invisible_claw_l2_reg/decoder_16_dim_3.0_beta.h5',
            'image_shape': (64, 64, 3),
            'latent_dim': 64,
            'encoder_path': os.path.join(NFS_PATH,
                                        'pretrained_models',
                                        'vae_64_dim_beta_5_visible_claw_diff_angle',
                                        'encoder_64_dim_5.0_beta.h5'),
            'trainable': False,
        },
    },
    'ConvnetPreprocessor': tune.grid_search([
        {
            'type': 'ConvnetPreprocessor',
            'kwargs': {
                'conv_filters': (16, 32, 64),
                'conv_kernel_sizes': (3, ) * 3,
                'conv_strides': (2, ) * 3,
                'normalization_type': normalization_type,
                'downsampling_type': 'conv',
                'output_kwargs': {
                    'type': 'flatten',
                }
            },
        }
        # {
        #     'type': 'ConvnetPreprocessor',
        #     'kwargs': {
        #         'conv_filters': (64, ) * 4,
        #         'conv_kernel_sizes': (3, ) * 4,
        #         'conv_strides': (2, ) * 4,
        #         'normalization_type': normalization_type,
        #         'downsampling_type': 'conv',
        #         'output_kwargs': {
        #             'type': 'flatten',
        #         },
        #     },
        #     # 'weights_path': '/root/nfs/kun1/users/justinvyu/pretrained_models/convnet_64_by_4.pkl',
        # }
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

    num_goals = 2
    variant_spec = {
        'git_sha': get_git_rev(),
        'num_goals': num_goals, # TODO: Separate classifier_rl with multigoal
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
                'kwargs': get_environment_params(universe, domain, task_eval, from_vision),
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
                'max_size': int(5e5),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                # 'max_path_length': get_max_path_length(universe, domain, task),
                # 'min_pool_size': get_max_path_length(universe, domain, task),
                'max_path_length': get_max_path_length(universe, domain, task),
                'min_pool_size': 50,
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

    # Filter out parts of the state relating to the object when training from pixels
    env_kwargs = variant_spec['environment_params']['training']['kwargs']
    if from_vision or ("pixel_wrapper_kwargs" in env_kwargs.keys() and \
       "device_path" not in env_kwargs.keys()):
        env_obs_keys = env_kwargs['observation_keys']

        non_image_obs_keys = tuple(key for key in env_obs_keys if key != 'pixels')
        variant_spec['replay_pool_params']['kwargs']['obs_save_keys'] = non_image_obs_keys

        non_object_obs_keys = tuple(key for key in env_obs_keys if 'object' not in key)
        variant_spec['policy_params']['kwargs']['observation_keys'] = variant_spec[
            'exploration_policy_params']['kwargs']['observation_keys'] = variant_spec[
            'Q_params']['kwargs']['observation_keys'] = non_object_obs_keys

    if 'Hardware' in task:
        env_kwargs['num_goals'] = num_goals
    return variant_spec


def get_variant_spec_classifier(universe,
                                domain,
                                task,
                                task_eval,
                                policy,
                                algorithm,
                                n_goal_examples,
                                from_vision,
                                *args,
                                **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, task_eval, policy, algorithm, from_vision, *args, **kwargs)

    variant_spec['reward_classifier_params'] = get_classifier_params(universe, domain, task)
    variant_spec['data_params'] = {
        'n_goal_examples': n_goal_examples,
        'n_goal_examples_validation_max': 100,
    }

    # variant_spec['sampler_params']['type'] = 'ClassifierSampler'
    # # Add classifier rewards to the replay pool
    # from softlearning.replay_pools.flexible_replay_pool import Field
    # variant_spec['replay_pool_params']['kwargs']['extra_fields'] = {
    #     'learned_rewards': Field(
    #         name='learned_rewards',
    #         dtype='float32',
    #         shape=(1, )
    #     )
    # }

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


CLASSIFIER_ALGS = (
    'SACClassifier',
    'RAQ',
    'VICE',
    'VICEGAN',
    'VICERAQ',
    'VICEGANTwoGoal',
    'VICEGANMultiGoal',
    'MultiVICEGAN'
)


def get_variant_spec(args):
    universe, domain = args.universe, args.domain
    task, task_eval, algorithm, n_epochs = (
        args.task, args.task_evaluation, args.algorithm, args.n_epochs)

    from_vision = args.vision

    if args.algorithm in CLASSIFIER_ALGS:
        variant_spec = get_variant_spec_classifier(
            universe, domain, task, task_eval, args.policy, algorithm,
            args.n_goal_examples, from_vision)
    else:
        variant_spec = get_variant_spec_base(
            universe, domain, task, task_eval, args.policy, algorithm, from_vision)

    if args.algorithm in ('RAQ', 'VICERAQ'):
        active_query_frequency = args.active_query_frequency
        variant_spec['algorithm_params']['kwargs'][
            'active_query_frequency'] = active_query_frequency

    variant_spec['algorithm_params']['kwargs']['n_epochs'] = n_epochs

    preprocessor_type = args.preprocessor_type

    if is_image_env(universe, domain, task, variant_spec):
        assert preprocessor_type in PIXELS_PREPROCESSOR_PARAMS
        preprocessor_params = PIXELS_PREPROCESSOR_PARAMS[preprocessor_type]

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
            reward_classifier_preprocessor_params = {
                'type': 'ConvnetPreprocessor',
                'kwargs': {
                    'conv_filters': (64, 64, 64),
                    'conv_kernel_sizes': (3, ) * 3,
                    'conv_strides': (2, 2, 2),
                    'normalization_type': None,
                    'downsampling_type': 'conv',
                    'output_kwargs': {
                        'type': 'flatten',
                    }
                },
            }
            (variant_spec
             ['reward_classifier_params']
             ['kwargs']
             ['observation_preprocessors_params']) = {
                'pixels': reward_classifier_preprocessor_params
            }
            # (variant_spec
            #  ['reward_classifier_params']
            #  ['kwargs']
            #  ['observation_preprocessors_params']) = (
            #     tune.sample_from(lambda spec: (
            #         spec.get('config', spec)
            #         ['policy_params']
            #         ['kwargs']
            #         ['observation_preprocessors_params']
            #     )))

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
