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

M = 256
# M = 128
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
    # 'rnd_params': {
    #     'convnet_params': {
    #         'conv_filters': (16, 32, 64),
    #         'conv_kernel_sizes': (3, 3, 3),
    #         'conv_strides': (2, 2, 2),
    #         'normalization_type': None,
    #     },
    #     'fc_params': {
    #         'hidden_layer_sizes': (256, 256),
    #         'output_size': 512,
    #     },
    # }
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

            'normalize_ext_reward_gamma': 0.99,
            'rnd_int_rew_coeff': tune.sample_from([0, 1]),
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
    'SQIL': {
        'type': 'SQIL',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
            'n_epochs': 200,
            'goal_negative_ratio': 1.0,
            'lambda_samp': 1.0,
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
            'n_classifier_train_steps': 2,
            'classifier_optim_name': 'adam',
            'n_epochs': 200,
            'mixup_alpha': tune.grid_search([1.]),
            'save_training_video_frequency': 0,

            # Tune over the reward scaling between count based bonus and VICE reward
            # 'ext_reward_coeff': tune.grid_search([0.25, 0.5, 1]),
            # 'normalize_ext_reward_gamma': tune.grid_search([0.99, 1]),
            # 'use_env_intrinsic_reward': tune.grid_search([True]),

            'positive_on_first_occurence': True,
        }
    },
    'VICEDynamicsAware': {
        'type': 'VICEDynamicsAware',
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
            'n_classifier_train_steps': 2,
            'classifier_optim_name': 'adam',
            'mixup_alpha': tune.grid_search([1.]),

            'train_dynamics_model_every_n_steps': tune.grid_search([16, 64]),
            'dynamics_model_lr': 3e-4,
            'dynamics_model_batch_size': 256,

            # 'normalize_ext_reward_gamma': tune.grid_search([0.99, 1]),
            # 'use_env_intrinsic_reward': tune.grid_search([True, False]),
        }
    },
    'MultiVICEGAN': {
        'type': 'MultiVICEGAN',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'her_iters': tune.grid_search([0]),
            # === BELOW FOR RND RESET CONTROLLER ===
            'rnd_int_rew_coeffs': tune.sample_from([[1, 1]]),
            'ext_reward_coeffs': [1, 0], # 0 corresponds to reset policy
            # === BELOW FOR 2 GOALS ===
            # 'rnd_int_rew_coeffs': tune.sample_from([[0, 0]]),
            # 'ext_reward_coeffs': [1, 1],
            'n_initial_exploration_steps': int(1e4),
            'normalize_ext_reward_gamma': 0.99,
            'share_pool': False,
            'n_classifier_train_steps': 5,
            'classifier_optim_name': 'adam',
            'n_epochs': 200,
            'mixup_alpha': 1.0,
            # 'eval_n_episodes': 24, # 24 for free screw
            'eval_n_episodes': 8, # 8 for beads, fixed screw
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
            'rnd_int_rew_coeff': tune.sample_from([5]),
            # Only train with RND reset controller
            'ext_reward_coeff': 0, # 0 corresponds to reset policy
            # 'normalize_ext_reward_gamma': 0.99,
            'n_initial_exploration_steps': int(1e4),
            'n_classifier_train_steps': 0,
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'classifier_optim_name': 'adam',
            'n_epochs': 1500,
            'mixup_alpha': 1.0,
            'normalize_ext_reward_gamma': 0.99,
            'rnd_int_rew_coeff': 0,
            'eval_n_episodes': 8,
        },
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
}

DEFAULT_NUM_EPOCHS = 200
NUM_CHECKPOINTS = 10

CLASSIFIER_PARAMS_BASE = {
    'type': 'feedforward_classifier',
    'kwargs': {
        'hidden_layer_sizes': (M, ) * N,
        'observation_keys': ('pixels', ),
    }
}

CLASSIFIER_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Point2D': {
            **{
                key: {'observation_keys': ('state_observation', )}
                for key in (
                    'Fixed-v0',
                    'SingleWall-v0',
                    'Maze-v0',
                    # 'BoxWall-v1',
                )
            },
            **{
                key: {'observation_keys': ('onehot_observation', )}
                for key in (
                    # 'Fixed-v0',
                    # 'SingleWall-v0',
                    # 'Maze-v0',
                    'BoxWall-v1',
                )
            },
        },
        'Pusher2D': {
            **{
                key: {
                    'observation_keys': tune.grid_search([
                        ('object_pos', ),
                        ('gripper_qpos', 'object_pos'),
                    ]),
                }
                for key in (
                    'Simple-v0',
                )
            },
        },
        'DClaw': {
            **{
                key: {'observation_keys': ('pixels', )}
                for key in (
                    'TurnResetFree-v0',
                    'TurnFreeValve3ResetFree-v0',
                    'SlideBeadsResetFree-v0',
                    'TurnFreeValve3Hardware-v0',
                )
            },
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

DYNAMICS_MODEL_PARAMS_BASE = {
    # 'type': 'feedforward_classifier',
    'kwargs': {
        'action_input': True,
        'encoder_kwargs': {
            'hidden_layer_sizes': (64, 64),
        },
        'decoder_kwargs': {
            'hidden_layer_sizes': (64, 64),
        },
        'dynamics_latent_dim': 16,
    }
}


ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_STATE = {
    'gym': {
        'Point2D': {
            # === Point Mass ===
            'Fixed-v0': {
                'action_scale': tune.grid_search([0.5]),
                'images_are_rgb': True,
                'init_pos_range': ((-2, -3), (-3, -3)), # Fixed reset
                # 'init_pos_range': None,             # Random reset
                'target_pos_range': ((3, 3), (3, 3)), # Set the goal to (x, y) = (2, 2)
                # 'target_pos_range': ((0, 0), (0, 0)), # Set the goal to (x, y) = (0, 0)
                'render_onscreen': False,
                'observation_keys': ('state_observation', ),
                'n_bins': 50,
                # 'observation_keys': ('onehot_observation', ),
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
                'init_pos_range': None,   # Random reset
                # 'target_pos_range': ((3.5, 3.5), (3.5, 3.5)),
                'target_pos_range': ((0, 3), (0, 3)),
                'render_onscreen': False,
                'observation_keys': ('onehot_observation', ),
                # 'observation_keys': ('state_observation', ),
            },
            'Maze-v0': {
                'action_scale': tune.grid_search([0.5]),
                'images_are_rgb': True,

                # === Use environment's count-based reward ===
                'reward_type': 'none',
                'use_count_reward': True,
                'n_bins': 50,  # Number of bins to discretize the space with

                # === EASY ===
                'wall_shape': 'easy-maze',
                'init_pos_range': ((-2.5, -3), (-2.5, -3)),
                'target_pos_range': ((2.5, -3), (2.5, -3)),
                # === MEDIUM ===
                # 'wall_shape': 'medium-maze',
                # 'init_pos_range': ((-3, -3), (-3, -3)),
                # 'target_pos_range': ((3, 3), (3, 3)),
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
            # === LIFTING === 
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
            },
            'TranslateMultiPuckFixed-v0': {
                'init_qpos_ranges': (
                    ((0.1, 0.1, 0, 0, 0, 0), (0.1, 0.1, 0, 0, 0, 0)),
                    ((-0.1, -0.1, 0, 0, 0, 0), (-0.1, -0.1, 0, 0, 0, 0)),
                ),
                'target_qpos_ranges': (
                    ((0.1, -0.1, 0, 0, 0, 0), (0.1, -0.1, 0, 0, 0, 0)),
                    ((-0.1, 0.1, 0, 0, 0, 0), (-0.1, 0.1, 0, 0, 0, 0)),
                ),
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'object1_xy_position',
                    'object2_xy_position',
                ),
                'reward_keys_and_weights': {
                    'object1_to_target_position_distance_log_reward': 1,
                    'object2_to_target_position_distance_log_reward': 1,
                }
            }
        }
    },
}


FREE_SCREW_VISION_KWARGS = {
    'pixel_wrapper_kwargs': {
        'pixels_only': False,
        'normalize': False,
        'render_kwargs': {
            'width': 32,
            'height': 32,
            'camera_id': -1,
        },
    },
    'camera_settings': {
        'azimuth': 180,
        'distance': 0.38,
        'elevation': -36,
        'lookat': (0.04, 0.008, 0.026),
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
SLIDE_BEADS_VISION_KWARGS = {
    'pixel_wrapper_kwargs': {
        'pixels_only': False,
        'normalize': False,
        'render_kwargs': {
            'width': 32,
            'height': 32,
            'camera_id': -1,
        },
    },
    'camera_settings': {
        'azimuth': 90,
        'distance': 0.37,
        'elevation': -45,
        'lookat': (0, 0.046, -0.016),
    },
}

ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_VISION = {
    'gym': {
        'DClaw': {
            # === FIXED SCREW RANDOM RESET EVAL TASK BELOW ===
            'TurnFixed-v0': {
                **FIXED_SCREW_VISION_KWARGS,
                # 'init_pos_range': (-np.pi, np.pi), # Random reset between -pi, pi
                # Reset to every 45 degrees between -pi and pi
                'init_pos_range': list(np.arange(-np.pi, np.pi, np.pi / 4)),

                # === GOAL = -90 DEGREES ===
                # Single goal + RND reset controller
                'target_pos_range': [-np.pi / 2, -np.pi / 2],
                # 2 goal + no RND reset controller
                # 'target_pos_range': [-np.pi / 2, np.pi / 2],
                # 1 goal + no RND reset controller
                # 'target_pos_range': [-np.pi / 2],
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
                'reset_fingers': True,
                'init_pos_range': (0, 0),
                # Single goal + RND reset controller
                'target_pos_range': [-np.pi / 2, -np.pi / 2],
                # 2 goal + no RND reset controller
                # 'target_pos_range': [-np.pi / 2, np.pi / 2],
                # 1 goal + no RND reset controller
                # 'target_pos_range': [-np.pi / 2],
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
                **FREE_SCREW_VISION_KWARGS,
                # Random init evaluations
                # 'init_qpos_range': (
                #     (-0.08, -0.08, 0, 0, 0, -np.pi),
                #     (0.08, 0.08, 0, 0, 0, np.pi)
                # ),
                # Evaluations from fixed set of inits
                'init_qpos_range': [
                    (0, 0, 0, 0, 0, 0),
                    (0, 0, 0, 0, 0, -np.pi),
                    (0, 0, 0, 0, 0, -np.pi / 2),
                    (0, 0, 0, 0, 0, np.pi / 2),
                    (-0.05, 0.075, 0, 0, 0, -np.pi),
                    (-0.075, 0.05, 0, 0, 0, -np.pi / 2),
                    (-0.05, 0.05, 0, 0, 0, -3 * np.pi / 4),
                    (-0.07, 0.07, 0, 0, 0, np.pi / 4),
                    (0, 0.075, 0, 0, 0, -np.pi),
                    (0.05, 0.075, 0, 0, 0, -np.pi),
                    (0.075, 0.05, 0, 0, 0, np.pi / 2),
                    (0.05, 0.05, 0, 0, 0, 3 * np.pi / 4),
                    (0.07, 0.07, 0, 0, 0, -np.pi / 4),
                    (-0.05, -0.075, 0, 0, 0, 0),
                    (-0.075, -0.05, 0, 0, 0, -np.pi / 2),
                    (-0.05, -0.05, 0, 0, 0, -np.pi / 4),
                    (-0.07, -0.07, 0, 0, 0, 3 * np.pi / 4),
                    (0, -0.075, 0, 0, 0, 0),
                    (0.05, -0.075, 0, 0, 0, 0),
                    (0.075, -0.05, 0, 0, 0, np.pi / 2),
                    (0.05, -0.05, 0, 0, 0, np.pi / 4),
                    (0.07, -0.07, 0, 0, 0, -3 * np.pi / 4),
                    (-0.075, 0, 0, 0, 0, -np.pi / 2),
                    (0.075, 0, 0, 0, 0, np.pi / 2),
                ],
                'cycle_inits': True,

                # 1 goal for RND reset controller
                # 'target_qpos_range': [
                #     (0, 0, 0, 0, 0, -np.pi / 2),
                #     (0, 0, 0, 0, 0, -np.pi / 2),
                # ],
                # 2 goal, no RND reset controller
                # 'target_qpos_range': [
                #     (0, 0, 0, 0, 0, -np.pi / 2),
                #     (0, 0, 0, 0, 0, np.pi / 2),
                # ],
                # 2 goals
                'target_qpos_range': [
                    # (top left, center)
                    # (-0.05, -0.05, 0, 0, 0, -np.pi / 2),
                    # (0, 0, 0, 0, 0, np.pi / 2),
                    # bottom right, top right
                    (0.075, 0.075, 0, 0, 0, -np.pi),
                    (-0.075, 0.075, 0, 0, 0, -np.pi)
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
                **FREE_SCREW_VISION_KWARGS,
                'init_qpos_range': [(0, 0, 0, 0, 0, 0)],
                # Below needs to be 2 for a MultiVICEGAN run, since the goals switch
                # Single goal + RND reset controller
                # 'target_qpos_range': [
                #     (0, 0, 0, 0, 0, -np.pi / 2),
                #     (0, 0, 0, 0, 0, -np.pi / 2), # Second goal is arbitrary
                # ],
                # 2 goal, no RND reset controller
                # 'target_qpos_range': [
                #     (0, 0, 0, 0, 0, -np.pi / 2),
                #     (0, 0, 0, 0, 0, np.pi / 2),
                # ],
                # 2 goals
                'target_qpos_range': [
                    # (top left, center)
                    # (-0.05, -0.05, 0, 0, 0, -np.pi / 2),
                    # (0, 0, 0, 0, 0, np.pi / 2),
                    # bottom right, top right
                    (0.075, 0.075, 0, 0, 0, -np.pi),
                    (-0.075, 0.075, 0, 0, 0, -np.pi)
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
            # === FREE SCREW HARDWARE ===
            'TurnFreeValve3Hardware-v0': {
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                    'normalize': False,
                    'render_kwargs': {
                       'width': 32,
                       'height': 32,
                       'camera_id': -1,
                       'box_warp': True,
                    }
                },
                'observation_keys': (
                    'claw_qpos',
                    'pixels',
                    'last_action',
                ),
                'device_path': '/dev/ttyUSB0',
                'camera_config': {
                    'topic': '/kinect2_001161563647/qhd/image_color',
                    'image_shape': (256, 256, 3),
                }
            },
            'TurnFreeValve3ResetFreeSwapGoal-v0': {
                **FREE_SCREW_VISION_KWARGS,
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
                **FREE_SCREW_VISION_KWARGS,
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
                **SLIDE_BEADS_VISION_KWARGS,
                'num_objects': 4,
                # Random init
                # 'init_qpos_range': (
                #     (-0.0475, -0.0475, -0.0475, -0.0475),
                #     (0.0475, 0.0475, 0.0475, 0.0475),
                # ),
                'init_qpos_range': [
                    (-0.0475, -0.0475, -0.0475, -0.0475), # 4 left
                    (0.0475, 0.0475, 0.0475, 0.0475), # 4 right
                    (0, 0, 0, 0), # 4 middle
                    (-0.0475, -0.0475, -0.0475, 0.0475), # 3 left, 1 right
                    (-0.0475, 0.0475, 0.0475, 0.0475), # 1 left, 3 right
                    (-0.0475, -0.0475, 0.0475, 0.0475), # 2 left, 2 right
                    (-0.0475, -0.02375, 0.02375, 0.0475), # even spaced
                    (-0.0475, 0, 0, 0.0475), # slides, and 2 in the middle
                ],
                'cycle_inits': True,
                # Goal we want to evaluate:
                'target_qpos_range': [
                    # 4 left
                    # (-0.0475, -0.0475, -0.0475, -0.0475),
                    # 2 left, 2 right
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                    # Remove below for 1 goal reset free
                    # (0, 0, 0, 0)
                ],
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'pixels',
                    # === BELOW JUST FOR LOGGING ===
                    'objects_positions',
                    'objects_target_positions',
                ),
            },
            'SlideBeadsResetFree-v0': {
                **SLIDE_BEADS_VISION_KWARGS,
                'init_qpos_range': [(0, 0, 0, 0)],
                'num_objects': 4,
                'target_qpos_range': [
                    # 4 left
                    # (-0.0475, -0.0475, -0.0475, -0.0475),
                    # 2 left, 2 right
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                    # This second one is arbitrary for training env
                    # (0, 0, 0, 0),
                ],
                'reset_fingers': False,
                'observation_keys': (
                    'pixels',
                    'claw_qpos',
                    'last_action',
                    # === BELOW JUST FOR LOGGING ===
                    'objects_target_positions',
                    'objects_positions',
                ),
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


def get_dynamics_model_params(universe, domain, task):
    params = DYNAMICS_MODEL_PARAMS_BASE.copy()
    # classifier_params['kwargs'].update(
    #     CLASSIFIER_PARAMS_PER_UNIVERSE_DOMAIN_TASK.get(
    #         universe, {}).get(domain, {}).get(task, {}))
    return params


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
            'trainable': False,
            # === Bead manipulation ===
            # 'image_shape': (32, 32, 3),
            # 'latent_dim': 16,
            # 'encoder_path': os.path.join(PROJECT_PATH,
            #                             'softlearning',
            #                             'models',
            #                             'slide_beads_vae_16_230iters',
            #                             'encoder_16_dim_1_beta.h5'),
            # === Free Screw ===
            'image_shape': (32, 32, 3),
            'latent_dim': 32, # 8,
            'encoder_path': os.path.join(PROJECT_PATH,
                                        'softlearning',
                                        'models',
                                        'hardware_free_screw_vae_black_box',
                                        'encoder_32_dim_0.5_beta_final.h5'),
            # 'encoder_path': os.path.join(PROJECT_PATH,
            #                             'softlearning',
            #                             'models',
            #                             'hardware_free_screw_vae_rnd_filtered_warped_include_claw',
            #                             'encoder_8_dim_0.5_beta_final.h5'),
            # === Fixed Screw ===
            # 'image_shape': (32, 32, 3),
            # 'latent_dim': 16,
            # 'encoder_path': os.path.join(PROJECT_PATH,
            #                             'softlearning',
            #                             'models',
            #                             'fixed_screw_16_dim_beta_half',
            #                             'encoder_16_dim_0.5_beta_210.h5')
        },
    },
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
            # Specify a `weights_path` here if you want to load in a pretrained convnet
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

    import tensorflow as tf
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
                )), # None means everything, pass in all keys but the goal_index
                'observation_preprocessors_params': {},
                'kernel_regularizer': tune.grid_search([
                    None,
                    # tf.keras.regularizers.l1(5e-4)
                ]),
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
    variant_spec['dynamics_model_params'] = get_dynamics_model_params(universe, domain, task)
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
    'VICEDynamicsAware',
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

    if algorithm in CLASSIFIER_ALGS:
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

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
