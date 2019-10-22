from copy import deepcopy

from ray import tune
import numpy as np
import os
from softlearning.misc.utils import get_git_rev, deep_update

DEFAULT_KEY = "__DEFAULT_KEY__"

# M = number of hidden units per layer
# N = number of hidden layers
# M = 256
# N = 2
M = 512
N = 3

REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2


GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, ) * N,
        'squash': True,
        'observation_keys': None,
        'observation_preprocessors_params': {}
    }
}


ALGORITHM_PARAMS_BASE = {
    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1, #tune.grid_search([1, 2, 5, 10]),
        'eval_n_episodes': 3, # num of eval rollouts
        'eval_deterministic': False,
        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
        'save_training_video_frequency': 5,
        'eval_render_kwargs': {
            'width': 480,
            'height': 480,
            'mode': 'rgb_array',
        },
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'n_initial_exploration_steps': int(5e3),
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'her_iters': tune.grid_search([0]),
            # === TO TRAIN ON RND REWARD ONLY ===
            'ext_reward_coeff': 1,
            # === DONT DO EVAL EPISODES FOR DATA COLLECTION ===
            'eval_n_episodes': 10,
            'rnd_int_rew_coeff': tune.sample_from([1]), # 1
            'normalize_ext_reward_gamma': 0.99,
            'online_vae': True, # False --> train at the end of each epoch
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
    'MultiSAC': {
        'type': 'MultiSAC',
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
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        }
    }
}


MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 1000,
    'gym': {
        DEFAULT_KEY: 1000,
        'Point2DEnv': {
            DEFAULT_KEY: 50,
        },
        'Pendulum': {
            DEFAULT_KEY: 300,
        },
        'Pusher2d': {
            DEFAULT_KEY: 300,
        },
        'InvisibleArm': {
            DEFAULT_KEY: 250,
        },
        'DClaw3': {
            DEFAULT_KEY: 250,
        },
        'HardwareDClaw3': {
            DEFAULT_KEY: 250,
        },
        'MiniGrid': {
            DEFAULT_KEY: 50,
        },
        'DClaw': {
            DEFAULT_KEY: 50,
            'TurnFixed-v0': 50,
            # 'TurnResetFree-v0': 100,
            'TurnResetFree-v0': 50,
            'TurnResetFreeSwapGoal-v0': tune.grid_search([100]),
            'TurnResetFreeRandomGoal-v0': 100,
            'TurnFreeValve3Fixed-v0': tune.grid_search([50]),
            # 'TurnFreeValve3RandomReset-v0': 50,
            'TurnFreeValve3ResetFree-v0': tune.grid_search([50]),
            'TurnFreeValve3ResetFreeSwapGoal-v0': tune.grid_search([50]),
            'TurnFreeValve3ResetFreeSwapGoalEval-v0': tune.grid_search([50]),
            'TurnFreeValve3ResetFreeComposedGoals-v0': tune.grid_search([150]),

            # Translating Tasks
            'TranslatePuckFixed-v0': 50,
            'TranslatePuckResetFree-v0': 50,

            # Lifting Tasks
            'LiftDDFixed-v0': tune.grid_search([50]),
            'LiftDDResetFree-v0': tune.grid_search([50]),

            # Flipping Tasks
            'FlipEraserFixed-v0': tune.grid_search([50]),
            'FlipEraserResetFree-v0': tune.grid_search([50]),
            'FlipEraserResetFreeSwapGoal-v0': tune.grid_search([50]),

            # Sliding Tasks
            'SlideBeadsFixed-v0': tune.grid_search([25]),
            'SlideBeadsResetFree-v0': tune.grid_search([25]),
            'SlideBeadsResetFreeEval-v0': tune.grid_search([25]),
        },
    },
}


NUM_EPOCHS_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 200,
    'gym': {
        DEFAULT_KEY: 200,
        'Swimmer': {
            DEFAULT_KEY: int(3e2),
        },
        'Hopper': {
            DEFAULT_KEY: int(1e3),
        },
        'HalfCheetah': {
            DEFAULT_KEY: int(3e3),
        },
        'Walker2d': {
            DEFAULT_KEY: int(3e3),
        },
        'Ant': {
            DEFAULT_KEY: int(3e3),
        },
        'Humanoid': {
            DEFAULT_KEY: int(1e4),
        },
        'Pusher2d': {
            DEFAULT_KEY: int(2e3),
        },
        'HandManipulatePen': {
            DEFAULT_KEY: int(1e4),
        },
        'HandManipulateEgg': {
            DEFAULT_KEY: int(1e4),
        },
        'HandManipulateBlock': {
            DEFAULT_KEY: int(1e4),
        },
        'HandReach': {
            DEFAULT_KEY: int(1e4),
        },
        'Point2DEnv': {
            DEFAULT_KEY: int(200),
        },
        'Reacher': {
            DEFAULT_KEY: int(200),
        },
        'Pendulum': {
            DEFAULT_KEY: 10,
        },
        'DClaw3': {
            DEFAULT_KEY: 200,
        },
        'HardwareDClaw3': {
            DEFAULT_KEY: 100,
        },
        'MiniGrid': {
            DEFAULT_KEY: 100,
        },
        'DClaw': {
            DEFAULT_KEY: int(1.5e3),
        },
    },
    'dm_control': {
        DEFAULT_KEY: 200,
        'ball_in_cup': {
            DEFAULT_KEY: int(2e4),
        },
        'cheetah': {
            DEFAULT_KEY: int(2e4),
        },
        'finger': {
            DEFAULT_KEY: int(2e4),
        },
    },
    'robosuite': {
        DEFAULT_KEY: 200,
        'InvisibleArm': {
            DEFAULT_KEY: int(1e3),
        },
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
                    'object_to_target_angle_dist_cost': 1,
                },
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'goal_index',
                    # 'one_hot_goal_index',
                    'object_angle_cos',
                    'object_angle_sin',
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
                    'object_to_target_angle_dist_cost': 1,
                },
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'goal_index',
                    'object_angle_cos',
                    'object_angle_sin',
                ),
            },

            'PoseStatic-v0': {},
            'PoseDynamic-v0': {},
            'TurnFixed-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_angle_distance_reward': 1,
                },
                'target_pos_range': (np.pi, np.pi),
                'init_pos_range': (-np.pi, np.pi),
            },
            'TurnRandom-v0': {},
            'TurnResetFree-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_angle_distance_reward': 1,
                },
                'reset_fingers': True,
                'init_pos_range': (0, 0),
                'target_pos_range': (np.pi, np.pi),
            },
            'TurnResetFreeSwapGoal-v0': {
                'reward_keys': (
                    'object_to_target_angle_dist_cost',
                ),
                'reset_fingers': True,
            },
            'TurnResetFreeRandomGoal-v0': {
                'reward_keys': (
                    'object_to_target_angle_dist_cost',
                ),
                'reset_fingers': True,
            },
            'TurnRandomDynamics-v0': {},
            'TurnFreeValve3Fixed-v0': {
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                    'target_xy_position',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                ),
                'target_qpos_range': [(0, 0, 0, 0, 0, 0)],
                'init_qpos_range': ((-0.08, -0.08, 0, 0, 0, -np.pi), (0.08, 0.08, 0, 0, 0, np.pi)),
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': tune.grid_search([2]),
                    'object_to_target_orientation_distance_reward': 1,
                },
            },
            'TurnFreeValve3ResetFree-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 0, # tune.sample_from([10]),
                    'object_to_target_orientation_distance_reward': 0, # 1,
                },
                'reset_fingers': True,
                'reset_frequency': 0,
                'reset_policy_checkpoint_path': '',
                'target_qpos_range': [
                    (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)
                ],
                'init_qpos_range': [(0, 0, 0, 0, 0, 0)],
                # === BELOW IS FOR SAVING INTO THE REPLAY POOL. ===
                # MAKE SURE TO SET `no_pixel_information = True` below in order
                # to remove the pixels from the policy inputs/Q inputs.
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                    },
                },
                'camera_settings': {
                    'azimuth': 180,
                    'distance': 0.38,
                    'elevation': -36,
                    'lookat': (0.04, 0.008, 0.025),
                },
                # 'camera_settings': {
                #     'azimuth': 180,
                #     'distance': 0.35,
                #     'elevation': -55,
                #     'lookat': (0, 0, 0.03),
                # },
                'observation_keys': (
                    'claw_qpos',
                    'object_xy_position',
                    'object_orientation_cos',
                    'object_orientation_sin',
                    'last_action',
                    'target_xy_position',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                    'pixels',
                ),
            },
            'TurnFreeValve3RandomReset-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                ),
                'initial_distribution_path': '',
                'reset_from_corners': True,
            },
            'TurnFreeValve3ResetFreeRandomGoal-v0': {
                'observation_keys': (
                    'claw_qpos',
                    'object_position',
                    'object_orientation_cos',
                    'object_orientation_sin',
                    'last_action',
                    'target_orientation',
                    'object_to_target_relative_position',
                ),
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                ),
                'reset_fingers': True,
            },
            'TurnFreeValve3ResetFreeSwapGoal-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': tune.grid_search([1, 2]),
                    'object_to_target_orientation_distance_reward': 1,
                    # 'object_to_target_position_distance_reward': tune.grid_search([1]),
                    # 'object_to_target_orientation_distance_reward': 0,
                },
                'reset_fingers': True,
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                    'target_xy_position',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                ),
                'goals': tune.grid_search([
                    [(0, 0, 0, 0, 0, np.pi / 2), (-0.05, -0.06, 0, 0, 0, 0)],
                    # [(0.05, 0.06, 0, 0, 0, 0), (-0.05, -0.06, 0, 0, 0, 0)],
                    # [(0, 0, 0, 0, 0, 0), (-0.05, -0.06, 0, 0, 0, 0)],
                ]),
            },
            'TurnFreeValve3ResetFreeSwapGoalEval-v0': {
                'reward_keys_and_weights': {
                    # 'object_to_target_position_distance_reward': tune.grid_search([1, 2]),
                    'object_to_target_position_distance_reward': tune.grid_search([2]),
                    'object_to_target_orientation_distance_reward': 1,
                },
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                    'target_xy_position',
                ),
                # 'goals': tune.grid_search([
                #     [(0, 0, 0, 0, 0, np.pi / 2), (-0.05, -0.06, 0, 0, 0, 0)],
                #     [(0.05, 0.06, 0, 0, 0, 0), (-0.05, -0.06, 0, 0, 0, 0)],
                #     [(0, 0, 0, 0, 0, 0), (-0.05, -0.06, 0, 0, 0, 0)],
                # ]),
            },
            'TurnFreeValve3ResetFreeCurriculum-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                ),
                'reset_fingers': False,
            },
            'XYTurnValve3Fixed-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                    'eef_to_object_xy_distance_cost',
                ),
            },
            'XYTurnValve3RandomReset-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                    'eef_to_object_xy_distance_cost',
                ),
                'num_goals': 1,
            },
            'XYTurnValve3Random-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                    'eef_to_object_xy_distance_cost',
                ),
            },
            'XYTurnValve3ResetFree-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                    'eef_to_object_xy_distance_cost',
                ),
                'reset_fingers': tune.grid_search([True, False]),
                'reset_arm': False,
            },
            'ScrewFixed-v0': {},
            'ScrewRandom-v0': {},
            'ScrewRandomDynamics-v0': {},

            # Lifting Tasks
            # 'LiftDDFixed-v0': {
            #     'reward_keys_and_weights': {
            #         'object_to_target_z_position_distance_reward': 10,
            #         'object_to_target_xy_position_distance_reward': 0,
            #         'object_to_target_orientation_distance_reward': 0, #5,
            #     },
            #     'init_qpos_range': [(0, 0, 0.041, 1.017, 0, 0)],
            #     'target_qpos_range': [(0, 0, 0.05, 0, 0, 0)]
            #     # [  # target pos relative to init
            #     #      (0, 0, 0, 0, 0, np.pi),
            #     #      (0, 0, 0, np.pi, 0, 0), # bgreen side up
            #     #      (0, 0, 0, 1.017, 0, 2*np.pi/5), # black side up
            #     #  ],
            # },
            'LiftDDFixed-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 10,
                    'object_to_target_xy_position_distance_reward': 0,
                    'object_to_target_orientation_distance_reward': 0, #5,
                },
                'init_qpos_range': (
                    (-0.05, -0.05, 0.041, -np.pi, -np.pi, -np.pi),
                    (0.05, 0.05, 0.041, np.pi, np.pi, np.pi)
                ),
                'target_qpos_range': (
                    (-0.05, -0.05, 0, 0, 0, 0),
                    (0.05, 0.05, 0, 0, 0, 0)
                ),
                'use_bowl_arena': False,
            },

            # 'LiftDDResetFree-v0': {
            #     'reward_keys_and_weights': {
            #         'object_to_target_z_position_distance_reward': 10,
            #         'object_to_target_xy_position_distance_reward': 0,
            #         'object_to_target_orientation_distance_reward': 0,
            #     },
            #     # 'init_qpos_range': [(0, 0, 0.041, 1.017, 0, 0)],
            #     'init_qpos_range': (
            #         (0, 0, 0.041, -np.pi, -np.pi, -np.pi),
            #         (0, 0, 0.041, np.pi, np.pi, np.pi),
            #     ),
            #     'target_qpos_range': [(0, 0, 0.05, 0, 0, 0)],
            # },
            'LiftDDResetFree-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 0,
                    'object_to_target_xy_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 0,
                },
                'init_qpos_range': (
                    (0, 0, 0.041, -np.pi, -np.pi, -np.pi),
                    (0, 0, 0.041, np.pi, np.pi, np.pi),
                ),
                'target_qpos_range': (
                    (-0.05, -0.05, 0, 0, 0, 0),
                    (0.05, 0.05, 0, 0, 0, 0)
                ),
                'use_bowl_arena': False,
            },

            # Flipping Tasks
            'FlipEraserFixed-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 20,
                },
                'init_qpos_range': [(0, 0, 0, 0, 0, 0)],
                'target_qpos_range': [(0, 0, 0, np.pi, 0, 0)],
            },
            'FlipEraserResetFree-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 20,
                },
            },
            'FlipEraserResetFreeSwapGoal-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 20,
                },
            },
        }
    }
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
        'lookat': (0, 0.0046, -0.016),
    },
}


ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_VISION = {
    'gym': {
        'swimmer': {  # 2 dof
        },
        'hopper': {  # 3 dof
        },
        'halfcheetah': {  # 6 dof
        },
        'walker2d': {  # 6 dof
        },
        'ant': {  # 8 dof
            'parameterizable-v3': {
                'healthy_reward': 0.0,
                'healthy_z_range': (-np.inf, np.inf),
                'exclude_current_positions_from_observation': False,
            }
        },
        'humanoid': {  # 17 dof
            'parameterizable-v3': {
                'healthy_reward': 0.0,
                'healthy_z_range': (-np.inf, np.inf),
                'exclude_current_positions_from_observation': False,
            }
        },
        'pusher2d': {  # 3 dof
            'default-v0': {
                'eef_to_puck_distance_cost_coeff': tune.grid_search([2.0]),
                'goal_to_puck_distance_cost_coeff': 1.0,
                'ctrl_cost_coeff': 0.0,
                #'goal': (0, -1),
                'puck_initial_x_range': (-1, 1), #(1, 1), #(0, 1),
                'puck_initial_y_range': (-1, 1), #(-0.5, -0.5), # (-1, -0.5),
                'goal_x_range': (-0.5, -0.5), #(-1, 0),
                'goal_y_range': (-0.5, -0.5), #(-1, 1),
                'num_goals': 2,
                'swap_goal_upon_completion': False,
                'reset_mode': "random",
                #'initial_distribution_path': "/mnt/sda/ray_results/gym/pusher2d/default-v0/2019-06-16t14-59-35-reset-free_single_goal_save_pool/experimentrunner_2_her_iters=0,n_initial_exploration_steps=2000,n_train_repeat=1,evaluation={'domain': 'pusher2d', 'task': 'defaul_2019-06-16_14-59-36umz5wb9o/",
                # 'pixel_wrapper_kwargs': {
                #     # 'observation_key': 'pixels',
                #     # 'pixels_only': true,
                #     'render_kwargs': {
                #         'width': 32,
                #         'height': 32,
                #         'camera_id': -1,
                #     },
                # },
            },
            'defaultreach-v0': {
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            },
            'imagedefault-v0': {
                'image_shape': (32, 32, 3),
                'arm_object_distance_cost_coeff': 0.0,
                'goal_object_distance_cost_coeff': 3.0,
            },
            'imagereach-v0': {
                'image_shape': (32, 32, 3),
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            },
            'blindreach-v0': {
                'image_shape': (32, 32, 3),
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            }
        },
        'Point2DEnv': {
            'Default-v0': {
                'observation_keys': ('observation', 'desired_goal'),
            },
            'Wall-v0': {
                'observation_keys': ('observation', 'desired_goal'),
            },
        },
        'Sawyer': {
            task_name: {
                'has_renderer': False,
                'has_offscreen_renderer': False,
                'use_camera_obs': False,
                'reward_shaping': tune.grid_search([True, False]),
            }
            for task_name in (
                    'Lift',
                    'NutAssembly',
                    'NutAssemblyRound',
                    'NutAssemblySingle',
                    'NutAssemblySquare',
                    'PickPlace',
                    'PickPlaceBread',
                    'PickPlaceCan',
                    'PickPlaceCereal',
                    'PickPlaceMilk',
                    'PickPlaceSingle',
                    'Stack',
            )
        },
        'DClaw': {
            'TurnFixed-v0': {
                **FIXED_SCREW_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'object_to_target_angle_distance_reward': 1,
                },
                'init_pos_range': (-np.pi, np.pi),
                'target_pos_range': [-np.pi / 2, -np.pi / 2],
                'observation_keys': (
                    'claw_qpos',
                    'pixels',
                    'last_action',
                    # 'target_angle_cos',
                    # 'target_angle_sin',
                    # === BELOW JUST FOR LOGGING ===
                    'object_angle_cos',
                    'object_angle_sin',
                ),
            },
            'TurnResetFree-v0': {
                **FIXED_SCREW_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'object_to_target_angle_distance_reward': 1,
                },
                'reset_fingers': True,
                'init_pos_range': (0, 0),
                'target_pos_range': [-np.pi / 2, -np.pi / 2],
                'observation_keys': (
                    'claw_qpos',
                    'pixels',
                    'last_action',
                    # 'target_angle_cos',
                    # 'target_angle_sin',
                    # === BELOW JUST FOR LOGGING ===
                    'object_angle_cos',
                    'object_angle_sin',
                ),
            },
            # Free screw
            'TurnFreeValve3Fixed-v0': {
                **FREE_SCREW_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 1,
                },
                'observation_keys': (
                    'pixels',
                    'claw_qpos',
                    'last_action',
                    # === BELOW JUST FOR LOGGING ===
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                ),
                'init_qpos_range': (
                    (-0.08, -0.08, 0, 0, 0, -np.pi),
                    (0.08, 0.08, 0, 0, 0, np.pi)
                ),
                'target_qpos_range': [
                    (0, 0, 0, 0, 0, -np.pi / 2),
                    (0, 0, 0, 0, 0, -np.pi / 2)
                ],
                # 'target_qpos_range': [
                #     (0, 0, 0, 0, 0, -np.pi / 2)
                # ],
            },
            # === Reset-free environment below ===
            'TurnFreeValve3ResetFree-v0': {
                **FREE_SCREW_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 1,
                },
                'reset_fingers': True,
                'reset_frequency': 0,
                'reset_policy_checkpoint_path': '',
                'init_qpos_range': [(0, 0, 0, 0, 0, 0)],
                'target_qpos_range': [
                    (0, 0, 0, 0, 0, -np.pi / 2),
                    (0, 0, 0, 0, 0, -np.pi / 2)
                ],
                'observation_keys': (
                    'pixels',
                    'claw_qpos',
                    'last_action',
                    # === BELOW JUST FOR LOGGING ===
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                ),
            },
            'TurnFreeValve3ResetFreeSwapGoal-v0': {
                **FREE_SCREW_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': tune.grid_search([0.5, 1]),
                    'object_to_target_orientation_distance_reward': 1,
                },
                'reset_fingers': True,
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'pixels',
                    'target_xy_position',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                    # === BELOW JUST FOR LOGGING ===
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                ),
            },
            'TurnFreeValve3ResetFreeSwapGoalEval-v0': {
                **FREE_SCREW_VISION_KWARGS,
                # 'reward_keys_and_weights': {
                #     'object_to_target_position_distance_reward': tune.grid_search([0.1, 0.5]),
                #     'object_to_target_orientation_distance_reward': 1,
                # },
                'init_qpos_range': (
                    (-0.08, -0.08, 0, 0, 0, -np.pi),
                    (0.08, 0.08, 0, 0, 0, np.pi)
                ),
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'pixels',
                    'target_xy_position',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                    # === BELOW JUST FOR LOGGING ===
                    'object_position',
                    'object_orientation_cos',
                    'object_orientation_sin',
                ),
            },
            'TurnFreeValve3RandomReset-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                ),
                'initial_distribution_path': '',
                'reset_from_corners': True,
            },
            'ScrewFixed-v0': {},
            'ScrewRandom-v0': {},
            'ScrewRandomDynamics-v0': {},
            # Translating Puck Tasks
            'TranslatePuckFixed-v0': {
                # 'pixel_wrapper_kwargs': {
                #     'observation_key': 'pixels',
                #     'pixels_only': False,
                #     'render_kwargs': {
                #         'width': 32,
                #         'height': 32,
                #     },
                # },
                # 'camera_settings': {
                #     'distance': 0.5,
                #     'elevation': -60
                # },
                # 'observation_keys': (
                #     'claw_qpos',
                #     'object_xy_position',
                #     'last_action',
                #     'target_xy_position',
                #     'pixels',
                # ),
                'target_qpos_range': [
                    (0, 0, 0, 0, 0, 0),
                    (0, 0, 0, 0, 0, 0)
                ],
                'init_qpos_range': (
                    (-0.08, -0.08, 0, 0, 0, 0),
                    (0.08, 0.08, 0, 0, 0, 0)
                ),
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                },
            },
            'TranslatePuckResetFree-v0': {
                # 'pixel_wrapper_kwargs': {
                #     'observation_key': 'pixels',
                #     'pixels_only': False,
                #     'render_kwargs': {
                #         'width': 32,
                #         'height': 32,
                #     },
                # },
                # 'camera_settings': {
                #     'distance': 0.5,
                #     'elevation': -60
                # },
                # 'observation_keys': (
                #     'claw_qpos',
                #     'object_xy_position',
                #     'last_action',
                #     'target_xy_position',
                #     'pixels',
                # ),
                'target_qpos_range': [
                    (-0.08, -0.08, 0, 0, 0, 0),
                    (0.08, 0.08, 0, 0, 0, 0)
                ],
                # 'init_qpos_range': (
                #     (-0.08, -0.08, 0, 0, 0, 0),
                #     (0.08, 0.08, 0, 0, 0, 0)
                # ),
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                },
            },

            # Lifting Tasks
            'LiftDDFixed-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 5,
                    'object_to_target_xy_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 0, #tune.sample_from([1, 5]), #5,
                },
                'init_qpos_range': (
                    (0, 0, 0.041, -np.pi, -np.pi, -np.pi),
                    (0, 0, 0.041, np.pi, np.pi, np.pi)
                ),
                'target_qpos_range': [(0, 0, 0.05, 0, 0, 0)],
                'use_bowl_arena': False,
                # 'target_qpos_range': [(0, 0, 0.0, 0, 0, np.pi)],
                # [  # target pos relative to init
                #      (0, 0, 0, 0, 0, np.pi),
                #      (0, 0, 0, np.pi, 0, 0), # bgreen side up
                #      (0, 0, 0, 1.017, 0, 2*np.pi/5), # black side up
                #  ],

                # 'pixel_wrapper_kwargs': {
                #     'observation_key': 'pixels',
                #     'pixels_only': False,
                #     'render_kwargs': {
                #         'width': 64,
                #         'height': 64,
                #     },
                # },
                # 'observation_keys': (
                #     'claw_qpos',
                #     'object_position',
                #     'object_quaternion',
                #     'last_action',
                #     'target_position',
                #     'target_quaternion',
                #     'pixels',
                # ),
                # 'camera_settings': {
                #     'azimuth': 180,
                #     'distance': 0.26,
                #     'elevation': -40,
                #     'lookat': (0, 0, 0.06),
                # },
                'reset_policy_checkpoint_path': '', #'/mnt/sda/ray_results/gym/DClaw/LiftDDFixed-v0/2019-08-01T18-06-55-just_lift_single_goal/id=3ac8c6e0-seed=5285_2019-08-01_18-06-565pn01_gq/checkpoint_1500/',
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
                        'width': 64,
                        'height': 64,
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
            'LiftDDResetFreeComposedGoals-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 1,
                    'object_to_target_xy_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 1, #tune.sample_from([1, 5]), #5,
                },
                'reset_policy_checkpoint_path': '',
                'goals': [
                     (0, 0, 0, 0, 0, 0),
                     (0, 0, 0.05, 0, 0, 0),
                     # (0, 0, 0, np.pi, 0, 0)
                 ],
                'reset_frequency': 0,
            },
            # Flipping Tasks
            'FlipEraserFixed-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 20,
                },
                # In bowl
                # 'camera_settings': {
                #     'azimuth': 180,
                #     'distance': 0.26,
                #     'elevation': -32,
                #     'lookat': (0, 0, 0.06)
                # },
                'observation_keys': (
                    'pixels', 'claw_qpos', 'last_action',
                    'object_position',
                    'object_quaternion',
                ),
                'reset_policy_checkpoint_path': None,
            },
            'LiftDDResetFree-v0': {
                # For repositioning
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 0,
                    'object_to_target_xy_position_distance_reward': 1,
                    'object_to_target_orientation_distance_reward': 0,
                },
                'init_qpos_range': (
                    (0, 0, 0.041, -np.pi, -np.pi, -np.pi),
                    (0, 0, 0.041, np.pi, np.pi, np.pi),
                ),
                'target_qpos_range': (
                    (-0.05, -0.05, 0, 0, 0, 0),
                    (0.05, 0.05, 0, 0, 0, 0)
                ),
                'use_bowl_arena': False,
                # For Lifting
                # 'reward_keys_and_weights': {
                #     'object_to_target_z_position_distance_reward': 10,
                #     'object_to_target_xy_position_distance_reward': tune.grid_search([1, 2]),
                #     'object_to_target_orientation_distance_reward': 0,
                # },
                # 'init_qpos_range': (
                #     (0, 0, 0.041, -np.pi, -np.pi, -np.pi),
                #     (0, 0, 0.041, np.pi, np.pi, np.pi),
                # ),
                # 'target_qpos_range': [(0, 0, 0.05, 0, 0, 0)],
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                    'normalize': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                        'camera_id': -1,
                    }
                },
                # In box
                'camera_settings': {
                    'azimuth': 180,
                    'distance': 0.35,
                    'elevation': -55,
                    'lookat': (0, 0, 0.03)
                },
                # In bowl
                # 'camera_settings': {
                #     'azimuth': 180,
                #     'distance': 0.26,
                #     'elevation': -32,
                #     'lookat': (0, 0, 0.06)
                # },
                'observation_keys': (
                    'pixels', 'claw_qpos', 'last_action',
                    'object_position',
                    'object_quaternion',
                ),
                'reset_policy_checkpoint_path': None,
            },
            # Sliding Tasks
            'SlideBeadsFixed-v0': {
                **SLIDE_BEADS_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'objects_to_targets_mean_distance_reward': 1,
                },
                'init_qpos_range': (
                    (-0.0475, -0.0475, -0.0475, -0.0475),
                    (0.0475, 0.0475, 0.0475, 0.0475),
                ),
                'target_qpos_range': [
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                ],
                'num_objects': 4,
                'cycle_goals': True,
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'pixels',
                    # === BELOW JUST FOR LOGGING == 
                    'objects_target_positions',
                    'objects_positions',
                ),
            },
            'SlideBeadsResetFree-v0': {
                **SLIDE_BEADS_VISION_KWARGS,
                'reward_keys_and_weights': {
                    'objects_to_targets_mean_distance_reward': 1,
                    # 'objects_to_targets_mean_distance_reward': 0, # Make sure 0 ext reward
                },
                'init_qpos_range': [(0, 0, 0, 0)],
                # LNT Baseline 
                # 'target_qpos_range': [
                    # (0, 0, 0, 0),
                    # (-0.0475, -0.0475, 0.0475, 0.0475),
                # ],
                # 1 goal with RND reset controller
                'target_qpos_range': [
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                    (-0.0475, -0.0475, 0.0475, 0.0475),
                ],
                'num_objects': 4,
                'cycle_goals': True,
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'pixels',
                    # === BELOW JUST FOR LOGGING ===
                    'objects_positions',
                    'objects_target_positions',
                ),
            },
            'SlideBeadsResetFreeEval-v0': {
                'reward_keys_and_weights': {
                    'objects_to_targets_mean_distance_reward': 1,
                },
                'init_qpos_range': [(0, 0, 0, 0)],
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
                    'azimuth': 23.234042553191497,
                    'distance': 0.2403358053524018,
                    'elevation': -29.68085106382978,
                    'lookat': (-0.00390331,  0.01236683,  0.01093447),
                }
            },
        },
    },
    'dm_control': {
        'ball_in_cup': {
            'catch': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                        'width': 84,
                        'height': 84,
                        'camera_id': 0,
                    },
                },
            },
        },
        'cheetah': {
            'run': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                        'width': 84,
                        'Height': 84,
                        'camera_id': 0,
                    },
                },
            },
        },
        'finger': {
            'spin': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                        'width': 84,
                        'height': 84,
                        'camera_id': 0,
                    },
                },
            },
        },
    },
}


def get_num_epochs(universe, domain, task):
    level_result = NUM_EPOCHS_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_max_path_length(universe, domain, task):
    level_result = MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_initial_exploration_steps(spec):
    config = spec.get('config', spec)
    initial_exploration_steps = 50 * (
        config
        ['sampler_params']
        ['kwargs']
        ['max_path_length']
    )

    return initial_exploration_steps


def get_checkpoint_frequency(spec):
    config = spec.get('config', spec)
    checkpoint_frequency = (
        config
        ['algorithm_params']
        ['kwargs']
        ['n_epochs']
    ) // NUM_CHECKPOINTS

    return checkpoint_frequency


def get_policy_params(universe, domain, task):
    policy_params = GAUSSIAN_POLICY_PARAMS_BASE.copy()
    return policy_params


def get_algorithm_params(universe, domain, task):
    algorithm_params = {
        'kwargs': {
            'n_epochs': get_num_epochs(universe, domain, task),
            'n_initial_exploration_steps': tune.sample_from(
                get_initial_exploration_steps),
            # 'n_initial_exploration_steps': 0,
        }
    }

    return algorithm_params


def get_environment_params(universe, domain, task, from_vision):
    if from_vision:
        params = ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_VISION
    else:
        params = ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK_STATE

    environment_params = (
        params.get(universe, {}).get(domain, {}).get(task, {}))

    return environment_params


NUM_CHECKPOINTS = 10
SAMPLER_PARAMS_PER_DOMAIN = {
    'DClaw': {
        'type': 'SimpleSampler',
    },
    'DClaw3': {
        'type': 'SimpleSampler',
    },
    'HardwareDClaw3': {
        'type': 'RemoteSampler',
    }
}


def get_variant_spec_base(universe, domain, task, task_eval, policy, algorithm, from_vision):
    algorithm_params = deep_update(
        ALGORITHM_PARAMS_BASE,
        get_algorithm_params(universe, domain, task),
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {}),
    )
    variant_spec = {
        'git_sha': get_git_rev(__file__),

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
        'policy_params': get_policy_params(universe, domain, task),
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
                'hidden_layer_sizes': (M, ) * N,
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                )),
                'observation_preprocessors_params': {}
            },
            # 'discrete_actions': False,
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': int(5e5),
            },
            'last_checkpoint_dir': '',
        },
        'sampler_params': deep_update({
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': get_max_path_length(universe, domain, task),
                'min_pool_size': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['sampler_params']['kwargs']['max_path_length']
                )),
                'batch_size': 256,
                'store_last_n_paths': 20,
            }
        }, SAMPLER_PARAMS_PER_DOMAIN.get(domain, {})),
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': tune.sample_from(get_checkpoint_frequency),
            'checkpoint_replay_pool': False,
        },
    }

    # Set this flag if you don't want to pass pixels into the policy/Qs
    no_pixel_information = False

    env_kwargs = variant_spec['environment_params']['training']['kwargs']
    if from_vision and "pixel_wrapper_kwargs" in env_kwargs.keys() and \
       "device_path" not in env_kwargs.keys():
        env_obs_keys = env_kwargs['observation_keys']
        # === UNCOMMENT BELOW IF YOU DONT WANT TO SAVE PIXELS INTO POOL ===
        non_image_obs_keys = tuple(key for key in env_obs_keys if key != 'pixels')
        variant_spec['replay_pool_params']['kwargs']['obs_save_keys'] = non_image_obs_keys

        non_object_obs_keys = tuple(key for key in env_obs_keys if 'object' not in key)
        variant_spec['policy_params']['kwargs']['observation_keys'] = variant_spec[
            'exploration_policy_params']['kwargs']['observation_keys'] = variant_spec[
                'Q_params']['kwargs']['observation_keys'] = non_object_obs_keys
    elif no_pixel_information:
        env_obs_keys = env_kwargs['observation_keys']
        non_pixel_obs_keys = tuple(key for key in env_obs_keys if key != 'pixels')
        variant_spec['policy_params']['kwargs']['observation_keys'] = variant_spec[
            'exploration_policy_params']['kwargs']['observation_keys'] = variant_spec[
                'Q_params']['kwargs']['observation_keys'] = non_pixel_obs_keys

    if 'ResetFree' not in task:
        variant_spec['algorithm_params']['kwargs']['save_training_video_frequency'] = 0
    # if task == 'TurnFreeValve3ResetFreeSwapGoal-v0':
    #     variant_spec['environment_params']['evaluation']['kwargs']['goals'] = (
    #         tune.sample_from(lambda spec: (
    #             spec.get('config', spec)
    #             ['environment_params']['training']['kwargs'].get('goals')
    #         ))
    #     )
    if domain == 'MiniGrid':
        variant_spec['algorithm_params']['kwargs']['reparameterize'] = False
        variant_spec['policy_params']['type'] = 'DiscretePolicy'
        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (32, 32)
        variant_spec['exploration_policy_params']['type'] = 'UniformDiscretePolicy'
        variant_spec['environment_params']['training']['kwargs']['normalize'] = False

    return variant_spec


IMAGE_ENVS = (
    ('robosuite', 'InvisibleArm', 'FreeFloatManipulation'),
)

def is_image_env(universe, domain, task, variant_spec):

    return ('image' in task.lower()
            or 'image' in domain.lower()
            or 'pixel_wrapper_kwargs' in (
                variant_spec['environment_params']['training']['kwargs'])
            or (universe, domain, task) in IMAGE_ENVS)

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
            'state_estimator_path': '/root/softlearning/softlearning/models/state_estimators/state_estimator_from_vae_latents.h5',
            'preprocessor_params': {
                'type': 'VAEPreprocessor',
                'kwargs': {
                    'encoder_path': '/root/softlearning/softlearning/models/vae_16_dim_beta_3_invisible_claw_l2_reg/encoder_16_dim_3.0_beta.h5',
                    'decoder_path': '/root/softlearning/softlearning/models/vae_16_dim_beta_3_invisible_claw_l2_reg/decoder_16_dim_3.0_beta.h5',
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
            'image_shape': (64, 64, 3),
            # 'latent_dim': 16,
            # 'encoder_path': '/nfs/kun1/users/justinvyu/pretrained_models/vae_16_dim_beta_3_invisible_claw_l2_reg/encoder_16_dim_3.0_beta.h5',
            # 'decoder_path': '/nfs/kun1/users/justinvyu/pretrained_models/vae_16_dim_beta_3_invisible_claw_l2_reg/decoder_16_dim_3.0_beta.h5',
            'latent_dim': 64,
            'encoder_path': os.path.join(NFS_PATH,
                                        'pretrained_models',
                                        'vae_64_dim_beta_5_visible_claw_diff_angle',
                                        'encoder_64_dim_5.0_beta.h5'),
            'trainable': False,
        },
    },
    # TODO: Merge OnlineVAEPreprocessor and VAEPreprocessor, just don't update
    # in SAC if not online
    'OnlineVAEPreprocessor': {
        'type': 'OnlineVAEPreprocessor',
        'kwargs': {
            'image_shape': (32, 32, 3),
            'latent_dim': 16,
            # 'latent_dim': 32,
            'beta': 0.5,
            # 'beta': 1e-5,
            # Optionally specify a pretrained model to start finetuning
            # 'encoder_path': os.path.join(PROJECT_PATH,
            #                              'softlearning',
            #                              'models',
            #                              'free_screw_vae_32_dim',
            #                              'encoder_32_dim_0.5_beta_final.h5'),
            # 'decoder_path': os.path.join(PROJECT_PATH,
            #                              'softlearning',
            #                              'models',
            #                              'free_screw_vae_32_dim',
            #                              'decoder_32_dim_0.5_beta_final.h5'),
        }
    },
    'RAEPreprocessor': {
        'type': 'RAEPreprocessor',
        'kwargs': {
            'image_shape': (32, 32, 3),
            'latent_dim': 32,
        }
    },
    'ConvnetPreprocessor': tune.grid_search([
        {
            'type': 'ConvnetPreprocessor',
            'kwargs': {
                'conv_filters': (8, 16, 32),
                'conv_kernel_sizes': (3, ) * 3,
                'conv_strides': (2, ) * 3,
                'normalization_type': tune.sample_from([None]),
                'downsampling_type': 'conv',
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

def get_variant_spec_image(universe,
                           domain,
                           task,
                           task_eval,
                           policy,
                           algorithm,
                           from_vision,
                           preprocessor_type,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe,
        domain,
        task,
        task_eval,
        policy,
        algorithm,
        from_vision,
        *args, **kwargs)

    if from_vision and is_image_env(universe, domain, task, variant_spec):
        assert preprocessor_type in PIXELS_PREPROCESSOR_PARAMS or preprocessor_type is None
        if preprocessor_type is None:
            preprocessor_type = "ConvnetPreprocessor"
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
    elif preprocessor_type is not None:
        # Assign preprocessor to all parts of the state
        assert preprocessor_type in STATE_PREPROCESSOR_PARAMS
        preprocessor_params = STATE_PREPROCESSOR_PARAMS[preprocessor_type]
        obs_keys = variant_spec['environment_params']['training']['kwargs']['observation_keys']

        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (M, ) * N
        variant_spec['policy_params']['kwargs'][
            'observation_preprocessors_params'] = {
                key: deepcopy(preprocessor_params)
                for key in obs_keys
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

    return variant_spec


def get_variant_spec(args):
    universe, domain, task, task_eval = (
        args.universe,
        args.domain,
        args.task,
        args.task_evaluation)

    from_vision = args.vision
    preprocessor_type = args.preprocessor_type

    variant_spec = get_variant_spec_image(
        universe,
        domain,
        task,
        task_eval,
        args.policy,
        args.algorithm,
        from_vision,
        preprocessor_type)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
