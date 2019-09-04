from copy import deepcopy

from ray import tune
import numpy as np
from softlearning.misc.utils import get_git_rev, deep_update

DEFAULT_KEY = "__DEFAULT_KEY__"

# M = number of hidden units per layer
# N = number of hidden layers
M = 256
N = 2

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
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000, #50,
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
            'target_entropy': 'auto', #tune.sample_from([-3, -5, -7]),#'auto',
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
            'her_iters': tune.grid_search([0]),
            # 'train_state_estimator_online': True,
            'train_state_estimator_online': False,
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
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
        'DClaw': {
            DEFAULT_KEY: 250,
            'TurnFixed-v0': 50,
            'TurnResetFree-v0': 100,
            'TurnResetFreeSwapGoal-v0': tune.grid_search([100]),
            'TurnResetFreeRandomGoal-v0': 100,
            'TurnFreeValve3Fixed-v0': tune.grid_search([50]),
            'TurnFreeValve3RandomReset-v0': 50,
            'TurnFreeValve3ResetFree-v0': tune.grid_search([50]),
            'TurnFreeValve3ResetFreeSwapGoal-v0': tune.grid_search([50]),
            'TurnFreeValve3ResetFreeSwapGoalEval-v0': tune.grid_search([50]),
            'TurnFreeValve3ResetFreeRandomGoal-v0': tune.grid_search([100]),
            'TurnFreeValve3FixedResetSwapGoal-v0': 50,
            'TurnRandomResetSingleGoal-v0': 100,
            'XYTurnValve3Fixed-v0': 50,
            'XYTurnValve3Random-v0': tune.grid_search([50, 100]),
            'XYTurnValve3RandomReset-v0': 100,
            'XYTurnValve3ResetFree-v0': 50,

            # Lifting Tasks
            'LiftDDFixed-v0': tune.grid_search([50]),
            'LiftDDResetFree-v0': tune.grid_search([50]),

            # Flipping Tasks
            'FlipEraserFixed-v0': tune.grid_search([50]),
            'FlipEraserResetFree-v0': tune.grid_search([50]),
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
            'PoseStatic-v0': {},
            'PoseDynamic-v0': {},
            'TurnFixed-v0': {
                'reward_keys': (
                    'object_to_target_angle_dist_cost',
                ),
                'init_object_pos_range': (0, 0),
                'target_pos_range': (np.pi, np.pi),
                'device_path': '/dev/ttyUSB0',

            },
            'TurnRandom-v0': {},
            'TurnResetFree-v0': {
                'reward_keys': (
                    'object_to_target_angle_dist_cost',
                ),
                'reset_fingers': True,
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
                'init_qpos_range': (
                    (-0.025, -0.025, 0, 0, 0, -np.pi),
                    (0.025, 0.025, 0, 0, 0, np.pi)
                ),
                'target_qpos_range': [(0, 0, 0, 0, 0, np.pi)],
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': tune.grid_search([1]),
                    'object_to_target_orientation_distance_reward': 1,
                },
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
                    # 'object_to_target_position_distance_reward': tune.grid_search([1, 2]),
                    'object_to_target_position_distance_reward': tune.grid_search([2]),
                    'object_to_target_orientation_distance_reward': 1,
                },
                'reset_fingers': True,
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
            'LiftDDFixed-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 10,
                    'object_to_target_xy_position_distance_reward': 0,
                    'object_to_target_orientation_distance_reward': 0, #5,
                },
                'init_qpos_range': [(0, 0, 0.041, 1.017, 0, 0)],
                'target_qpos_range': [(0, 0, 0.05, 0, 0, 0)]
                # [  # target pos relative to init
                #      (0, 0, 0, 0, 0, np.pi),
                #      (0, 0, 0, np.pi, 0, 0), # bgreen side up
                #      (0, 0, 0, 1.017, 0, 2*np.pi/5), # black side up
                #  ],
            },
            'LiftDDResetFree-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 10,
                    'object_to_target_xy_position_distance_reward': 0,
                    'object_to_target_orientation_distance_reward': 0,
                },
                # 'init_qpos_range': [(0, 0, 0.041, 1.017, 0, 0)],
                'init_qpos_range': (
                    (0, 0, 0.041, -np.pi, -np.pi, -np.pi),
                    (0, 0, 0.041, np.pi, np.pi, np.pi),
                ),
                'target_qpos_range': [(0, 0, 0.05, 0, 0, 0)],
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
            'TurnRandomResetSingleGoal-v0': {
                'initial_object_pos_range': (-np.pi, np.pi),
                'reward_keys': ('object_to_target_angle_dist_cost', ),
                'device_path': '/dev/ttyUSB2',
                'camera_config': {
                    # 'topic': '/kinect2_001144463747/hd/image_color',
                    'topic': '/front_2/image_raw',
                },
                'use_dict_obs': True,
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': False,
                    'render_kwargs': {
                        'width': 32,
                        'height': 32,
                        'camera_id': 1,
                        # 'camera_name': 'track',
                    },
                },
                'observation_keys': ('claw_qpos', 'last_action', 'pixels'),
            },
            'TurnFreeValve3Fixed-v0': {
                'camera_settings': {
                    'azimuth': 180,
                    'distance': 0.35,
                    'elevation': -55,
                    'lookat': (0, 0, 0.03),
                },
                'pixel_wrapper_kwargs': {
                    'pixels_only': False,
                    'normalize': False,
                    'render_kwargs': {
                        'camera_id': -1,
                        'width': 32,
                        'height': 32
                    }
                },
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'pixels',
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                    'target_xy_position',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                ),
                'init_qpos_range': (
                    (-0.025, -0.025, 0, 0, 0, -np.pi),
                    (0.025, 0.025, 0, 0, 0, np.pi)
                ),
                'target_qpos_range': [(0, 0, 0, 0, 0, np.pi)],
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': tune.grid_search([0.1, 0.5]),
                    # 'object_to_target_position_distance_reward': tune.grid_search([0.1]),
                    'object_to_target_orientation_distance_reward': 1,
                },
            },
            'TurnFreeValve3ResetFree-v0': {
                'reward_keys': (
                    'object_to_target_position_distance_cost',
                    'object_to_target_orientation_distance_cost',
                ),
                'reset_fingers': True,
                'position_reward_weight': tune.sample_from([1]),
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
                    # 'object_orientation_cos',
                    # 'object_orientation_sin',
                    'last_action',
                    'target_orientation',
                    #    'target_orientation_cos',
                    #    'target_orientation_sin',
                    # 'object_to_target_relative_position',
                    #    'in_corner',
                    'pixels',
                ),

            },
            'TurnFreeValve3ResetFreeRandomGoal-v0': {
            },
            'TurnFreeValve3ResetFreeSwapGoal-v0': {
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
                    'distance': 0.35,
                    'elevation': -55,
                    'lookat': (0, 0, 0.03),
                },
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': tune.grid_search([0.1, 0.5]),
                    # 'object_to_target_position_distance_reward': 0.5,
                    'object_to_target_orientation_distance_reward': 1,
                },
                'reset_fingers': True,
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'pixels',
                    'object_xy_position',
                    'object_z_orientation_cos',
                    'object_z_orientation_sin',
                    'target_xy_position',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                ),
            },
            'TurnFreeValve3ResetFreeSwapGoalEval-v0': {
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
                    'distance': 0.35,
                    'elevation': -55,
                    'lookat': (0, 0, 0.03)
                },
                'reward_keys_and_weights': {
                    'object_to_target_position_distance_reward': tune.grid_search([0.5, 1]),
                    # 'object_to_target_position_distance_reward': 2,
                    'object_to_target_orientation_distance_reward': 1,
                },
                'observation_keys': (
                    'claw_qpos',
                    'last_action',
                    'pixels',
                    'object_position',
                    'object_orientation_cos',
                    'object_orientation_sin',
                    'target_xy_position',
                    'target_z_orientation_cos',
                    'target_z_orientation_sin',
                ),
            },
            # Lifting Tasks
            'LiftDDFixed-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 10,
                    'object_to_target_xy_position_distance_reward': 0,
                    'object_to_target_orientation_distance_reward': 0,
                },
                'init_qpos_range': (
                    (0, 0, 0.041, -np.pi, -np.pi, -np.pi),
                    (0, 0, 0.041, np.pi, np.pi, np.pi),
                ),
                # 'init_qpos_range': (
                #     (-0.04, -0.04, 0.041, 1.017, 0, 0),
                #     (0.04, 0.04, 0.041, 1.017, 0, 0)
                # ),
                'target_qpos_range': [(0, 0, 0.05, 0, 0, 0)],
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
                    'distance': 0.26,
                    'elevation': -32,
                    'lookat': (0, 0, 0.06)
                },
                'observation_keys': (
                    'pixels', 'claw_qpos', 'last_action',
                    'object_position',
                    'object_quaternion',
                ),
                'reset_policy_checkpoint_path': None,
            },
            'LiftDDResetFree-v0': {
                'reward_keys_and_weights': {
                    'object_to_target_z_position_distance_reward': 10,
                    'object_to_target_xy_position_distance_reward': tune.grid_search([1, 2]),
                    'object_to_target_orientation_distance_reward': 0,
                },
                'init_qpos_range': (
                    (0, 0, 0.041, -np.pi, -np.pi, -np.pi),
                    (0, 0, 0.041, np.pi, np.pi, np.pi),
                ),
                'target_qpos_range': [(0, 0, 0.05, 0, 0, 0)],
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
                    'distance': 0.26,
                    'elevation': -32,
                    'lookat': (0, 0, 0.06)
                },
                'observation_keys': (
                    'pixels', 'claw_qpos', 'last_action',
                    'object_position',
                    'object_quaternion',
                ),
                'reset_policy_checkpoint_path': None,
            },

            # Flipping Tasks
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
                        'height': 84,
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


def evaluation_environment_params(spec):
    training_environment_params = (spec.get('config', spec)
                                   ['environment_params']
                                   ['training'])
    eval_environment_params = training_environment_params.copy()
    if training_environment_params['task'] == 'TurnFreeValve3ResetFree-v0':
        eval_environment_params['task'] = 'TurnFreeValve3Fixed-v0'
        eval_environment_params['kwargs'] = {
            'reward_keys': (
                'object_to_target_position_distance_cost',
                'object_to_target_orientation_distance_cost',
            ),
            # 'initial_distribution_path': '/mnt/sda/ray_results/gym/DClaw/TurnFreeValve3ResetFree-v0/2019-06-30T18-53-06-baseline_both_push_and_turn_log_rew/id=38872574-seed=6880_2019-06-30_18-53-07whkq1aax/',
            # 'reset_from_corners': False,
        }
        pass
    elif training_environment_params['task'] == 'TurnFreeValve3ResetFreeSwapGoal-v0':
        eval_environment_params['task'] = 'TurnFreeValve3ResetFreeSwapGoalEval-v0' #'TurnFreeValve3RandomReset-v0'
        eval_environment_params['kwargs'].update({
            # 'initial_distribution_path': '/mnt/sda/ray_results/gym/DClaw/TurnFreeValve3ResetFree-v0/2019-06-30T18-53-06-baseline_both_push_and_turn_log_rew/id=38872574-seed=6880_2019-06-30_18-53-07whkq1aax/',
            # 'reset_from_corners': False,
        })
        eval_environment_params['kwargs'].pop('reset_fingers')

    elif training_environment_params['task'] == 'TurnFreeValve3ResetFreeCurriculum-v0':
        eval_environment_params['task'] = 'TurnFreeValve3ResetFreeCurriculumEval-v0' #'TurnFreeValve3RandomReset-v0'
        eval_environment_params['kwargs'] = {
            'reward_keys': (
                'object_to_target_position_distance_cost',
                'object_to_target_orientation_distance_cost',
            ),
            # 'initial_distribution_path': '/mnt/sda/ray_results/gym/DClaw/TurnFreeValve3ResetFree-v0/2019-06-30T18-53-06-baseline_both_push_and_turn_log_rew/id=38872574-seed=6880_2019-06-30_18-53-07whkq1aax/',
            # 'reset_from_corners': False,
        }
    elif training_environment_params['task'] == 'FlipEraserResetFreeSwapGoal-v0':
        eval_environment_params['task'] = 'FlipEraserResetFreeSwapGoalEval-v0' #'TurnFreeValve3RandomReset-v0'
        eval_environment_params['kwargs'] = {
            'reward_keys_and_weights': {
                'object_to_target_position_distance_reward': 1,
                'object_to_target_orientation_distance_reward': 20,
            },
        }

    return eval_environment_params


def get_variant_spec_base(universe, domain, task, policy, algorithm, from_vision):
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
            'evaluation': tune.sample_from(lambda spec: evaluation_environment_params(spec)),
            #     spec.get('config', spec)
            #     ['environment_params']
            #     ['training']
            # )),
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
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': int(1.5e6),
            },
            'last_checkpoint_dir': '',
            # 'last_checkpoint_dir': '/home/justinvyu/ray_results/gym/DClaw/TurnFreeValve3ResetFreeSwapGoal-v0/2019-08-07T14-57-41-state_gtr_2_goals_with_resets_regular_box_saving_pixels_fixed_env/id=612875d0-seed=9463_2019-08-07_14-57-42op75_8n7',
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

    if task == 'InfoScrewV2-v0':
        variant_spec['replay_pool_params']['kwargs']['include_images'] = True
    if task == 'ImageScrewV2-v0' and ENVIRONMENT_PARAMS['DClaw3']['ImageScrewV2-v0']['state_reward']:
        variant_spec['replay_pool_params']['kwargs']['super_observation_space_shape'] = (9+9+2+1+2,)
    if domain == 'HardwareDClaw3':
        variant_spec['sampler_params']['type'] == 'RemoteSampler'
        variant_spec['algorithm_params']['kwargs']['max_train_repeat_per_timestep'] = 1
    if task == 'TurnFreeValve3ResetFree-v0':
        pass
        # variant_spec['replay_pool_params']['type'] = 'PartialSaveReplayPool'
        # variant_spec['replay_pool_params']['kwargs']['mode'] = 'Bellman_Error'
        # variant_spec['replay_pool_params']['kwargs']['per_alpha'] = tune.grid_search([1, 0.1, 0.5, 1])
        # DEFAULT_OBSERVATION_KEYS = (
        #     'claw_qpos',
        #     'object_position',
        #     'object_orientation_cos',
        #     'object_orientation_sin',
        #     'last_action',
        #     'target_orientation_cos',
        #     'target_orientation_sin',
        #     'object_to_target_relative_position',
        # )
    if task == 'TurnFreeValve3ResetFreeSwapGoal-v0':
        pass
        # variant_spec['replay_pool_params']['type'] = 'PrioritizedExperienceReplayPool'
        # variant_spec['replay_pool_params']['kwargs']['mode'] = 'Bellman_Error'
        # variant_spec['replay_pool_params']['kwargs']['per_alpha'] = tune.grid_search([0.25, 0.5, 0.75])
        # DEFAULT_OBSERVATION_KEYS = (
        #     'claw_qpos',
        #     'object_position',
        #     'object_orientation_cos',
        #     'object_orientation_sin',
        #     'last_action',
        #     'target_orientation_cos',
        #     'target_orientation_sin',
        #     'object_to_target_relative_position',
        # )
        # variant_spec['environment_params']['training']['kwargs'][
        #     'observation_keys'] = DEFAULT_OBSERVATION_KEYS + ('in_corner', 'other_reward')
        # variant_spec['policy_params']['kwargs']['observation_keys'] = variant_spec[
        #     'exploration_policy_params']['kwargs']['observation_keys'] = variant_spec[
        #         'Q_params']['kwargs']['observation_keys'] = DEFAULT_OBSERVATION_KEYS

    # variant_spec['replay_pool_params']['type'] = 'UniformlyReweightedReplayPool'
    # variant_spec['replay_pool_params']['kwargs'].update({
    #     'bin_boundaries': (
    #         np.arange(-0.1, 0.1, 0.01),
    #         np.arange(-0.1, 0.1, 0.01),
    #         np.arange(-np.pi, np.pi, 0.2),
    #     ),
    #     'bin_keys': (
    #         ('infos', 'obs/object_xy_position'),
    #         ('infos', 'obs/object_angle'),
    #     ),
    #     'bin_weight_bonus_scaling': 1000,
    # })

    # TODO: Add this as a command line arg
    no_pixel_information = False

    env_kwargs = variant_spec['environment_params']['training']['kwargs']
    if from_vision and "pixel_wrapper_kwargs" in env_kwargs.keys() and \
       "device_path" not in env_kwargs.keys():
        env_obs_keys = env_kwargs['observation_keys']
        non_image_obs_keys = tuple(key for key in env_obs_keys if key != 'pixels')
        # variant_spec['replay_pool_params']['kwargs']['obs_save_keys'] = non_image_obs_keys

        non_object_obs_keys = tuple(key for key in env_obs_keys if 'object' not in key)
        variant_spec['policy_params']['kwargs']['observation_keys'] = variant_spec[
            'exploration_policy_params']['kwargs']['observation_keys'] = variant_spec[
                'Q_params']['kwargs']['observation_keys'] = non_object_obs_keys
    elif no_pixel_information:
        non_pixel_obs_keys = tuple(key for key in env_obs_keys if key != 'pixels')
        variant_spec['policy_params']['kwargs']['observation_keys'] = variant_spec[
            'exploration_policy_params']['kwargs']['observation_keys'] = variant_spec[
                'Q_params']['kwargs']['observation_keys'] = non_pixel_obs_keys

    if 'ResetFree' not in task:
        variant_spec['algorithm_params']['kwargs']['save_training_video_frequency'] = 0

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
}

PIXELS_PREPROCESSOR_PARAMS = {
    'StateEstimatorPreprocessor': {
        'type': 'StateEstimatorPreprocessor',
        'kwargs': {
            # 'domain': domain,
            # 'task': task,
            # 'obs_keys_to_estimate': (
            #     'object_position',
            #     'object_orientation_cos',
            #     'object_orientation_sin',
            # ),
            'input_shape': (32, 32, 3),
            'num_hidden_units': 256,
            'num_hidden_layers': 2,
            'state_estimator_path': '/home/justinvyu/dev/softlearning-vice/softlearning/models/state_estimators/state_estimator_fixed_antialias.h5'
        }
    },
    'VAEPreprocessor': {
        'type': 'VAEPreprocessor',
        'kwargs': {
            'image_shape': (32, 32, 3),
            'latent_dim': 16,
            'encoder_path': '/root/softlearning/softlearning/models/vae_weights/invisible_claw_encoder_weights.h5',
            'decoder_path': '/root/softlearning/softlearning/models/vae_weights/invisible_claw_decoder_weights.h5',
        },
    },
    'ConvnetPreprocessor': tune.grid_search([
        {
            'type': 'ConvnetPreprocessor',
            'kwargs': {
                'conv_filters': (16, 32, 64, 32),
                'conv_kernel_sizes': (3, ) * 4,
                'conv_strides': (2, 2, 1, 1),
                'normalization_type': normalization_type,
                'downsampling_type': 'conv',
                'output_kwargs': {
                    'type': 'spatial_softmax',
                }
            },
        }
        for normalization_type in (None, )
    ]),
}

def get_variant_spec_image(universe,
                           domain,
                           task,
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
        policy,
        algorithm,
        from_vision,
        *args, **kwargs)

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
    else:
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
    universe, domain, task = args.universe, args.domain, args.task

    from_vision = args.vision
    preprocessor_type = args.preprocessor_type

    variant_spec = get_variant_spec_image(
        universe,
        domain,
        task,
        args.policy,
        args.algorithm,
        from_vision,
        preprocessor_type)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
