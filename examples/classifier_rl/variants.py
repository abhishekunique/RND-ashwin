from copy import deepcopy

from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update
from softlearning.misc.generate_goal_examples import DOOR_TASKS, PUSH_TASKS, PICK_TASKS

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
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

DEFAULT_MAX_PATH_LENGTH = 200
MAX_PATH_LENGTH_PER_DOMAIN = {}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_kwargs': {},
        'eval_n_episodes': 5,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
    }
}

#TODO Avi Most of the algorithm params for classifier-style methods
#are shared. Rewrite this part to reuse the params
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
            #'target_entropy': tune.grid_search([-10, -7, -5, -2, 0, 5]),
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
            'n_classifier_train_steps': 10,
            'classifier_optim_name': 'adam',
            'n_epochs': 200,
            'mixup_alpha': 1.0,
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

ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'DClaw': {
            # Add in tasks
            'TurnImage-v0': {
                'observation_keys': ('image',),
                'image_shape': (32, 32, 3),
                'camera_settings': {
                    'azimuth': 90.18582,
                    'distance': 0.32,
                    'elevation': -32.42,
                    'lookat': np.array([-0.00157929, 0.00336185, 0.10151641])
                },
                'init_angle_range': (0., 0.),
                'target_angle_range': (np.pi, np.pi),
            },
        },
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

def get_environment_params(universe, domain, task):
    environment_params = (
        ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK
        .get(universe, {}).get(domain, {}).get(task, {}))
    return environment_params

"""
Configuring variant specs
"""

def get_variant_spec_base(universe, domain, task, policy, algorithm):
    # algorithm_params = deep_update(
    #     ALGORITHM_PARAMS_BASE,
    #     ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
    # )
    # algorithm_params = deep_update(
    #     algorithm_params,
    #     ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    # )
    algorithm_params = ALGORITHM_PARAMS_BASE
    algorithm_params = deep_update(
            algorithm_params,
            ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
        )

    variant_spec = {
        'git_sha': get_git_rev(__file__),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': get_environment_params(universe, domain, task),
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
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
                'hidden_layer_sizes': (M, M),
                'observation_keys': None,
                'observation_preprocessors_params': {}
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': int(1e6)
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': get_max_path_length(universe, domain, task),
                'min_pool_size': get_max_path_length(universe, domain, task),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': tune.sample_from(get_checkpoint_frequency),
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec


def get_variant_spec_classifier(universe,
                                domain,
                                task,
                                policy,
                                algorithm,
                                n_goal_examples,
                                *args,
                                **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    classifier_layer_size = L = 256
    variant_spec['reward_classifier_params'] = {
        'type': 'feedforward_classifier',
        'kwargs': {
            'hidden_layer_sizes': (L,L),
            }
        }

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
                    'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH),
                    'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH),
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

def get_variant_spec(args):
    universe, domain = args.universe, args.domain
    task, algorithm, n_epochs = args.task, args.algorithm, args.n_epochs
    active_query_frequency = args.active_query_frequency

    if args.algorithm in ['SACClassifier', 'RAQ', 'VICE', 'VICEGAN', 'VICERAQ']:
        variant_spec = get_variant_spec_classifier(
            universe, domain, task, args.policy, args.algorithm,
            args.n_goal_examples)
    else:
        variant_spec = get_variant_spec_base(
            universe, domain, task, args.policy, args.algorithm)

    if args.algorithm in ['RAQ', 'VICERAQ']:
        variant_spec['algorithm_params']['kwargs']['active_query_frequency'] = \
            active_query_frequency

    variant_spec['algorithm_params']['kwargs']['n_epochs'] = n_epochs

    if 'Image' in task:
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'conv_filters': (64, ) * 3,
                'conv_kernel_sizes': (3, ) * 3,
                'conv_strides': (2, ) * 3,
                'normalization_type': 'layer',
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

        if args.algorithm in ['SACClassifier', 'RAQ', 'VICE', 'VICEGAN', 'VICERAQ']:
            (variant_spec
             ['reward_classifier_params']
             ['kwargs']
             ['observation_preprocessors_params']) = (
                 preprocessor_params.copy())

    elif 'Image' in task:
        raise NotImplementedError('Add convnet preprocessor for this image input')

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
