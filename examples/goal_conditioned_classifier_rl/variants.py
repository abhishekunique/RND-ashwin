from copy import deepcopy

from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update
from softlearning.misc.generate_goal_examples import DOOR_TASKS, PUSH_TASKS, PICK_TASKS
from softlearning.replay_pools.hindsight_experience_replay_pool import REPLACE_FLAT_OBSERVATION
from softlearning.algorithms.sac_classifier import SACClassifier

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
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
    'Point2DEnv': 50,
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_kwargs': {'mode' : 'rgb_array'},
        'eval_n_episodes': 3,
        'eval_deterministic': False,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
        'save_training_video': False,
    }
}

# TODO(Avi) Most of the algorithm params for classifier-style methods
# are shared. Rewrite this part to reuse the params.
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
    'VICEGoalConditioned': {
        'type': 'VICEGoalConditioned',
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
            'n_classifier_train_steps': 25,
            'classifier_optim_name': 'adam',
            'n_epochs': 200,
            'mixup_alpha': 1.0,
        }
    },
    'VICEGANGoalConditioned': {
        'type': 'VICEGANGoalConditioned',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            # 'lr':  tune.grid_search([3e-4, 1e-3]),
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 25,
            'classifier_optim_name': 'adam',
            'n_epochs': 200,
            'mixup_alpha': 1.0,
            'hindsight_goal_prob': tune.grid_search([0.]), # 0.5, 0.8
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
Additional environment params
"""
ENV_PARAMS = {
    'DClaw': {
        'TurnImageMultiGoalResetFree-v0': {
            'initial_goal_index': 0,
            'goal_image_pools_path': '/Users/justinvyu/Developer/summer-2019/goal-conditioned-vice/goal_pools/fixed_screw_multigoal_0_180/positives.pkl',
            'swap_goals_upon_completion': True,
            'pixel_wrapper_kwargs': {
                'pixels_only': False,
                # Free camera
                'render_kwargs': {
                    'width': 32, 'height': 32, 'camera_id': -1
                }
            },
            'observation_keys': ('pixels',) #'claw_qpos', 'last_action')
        }
    }
}

def get_variant_spec_base(universe, domain, task, task_evaluation, policy, algorithm):
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
        'domain': domain,
        'task': task,
        'task_evaluation': task_evaluation,
        'universe': universe,
        'git_sha': get_git_rev(),

        'env_params': ENV_PARAMS.get(domain, {}).get(task, {}),
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
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            # 'type': 'SimpleReplayPool',
            # 'type': 'RelabelReplayPool',
            'type': 'HindsightExperienceReplayPool',
            'kwargs': {
                'max_size': 200000,
                # implement this
                'update_batch_fn': tune.function(REPLACE_FLAT_OBSERVATION),
                #'reward_fn': tune.function(SACClassifier._reward_relabeler),
                'reward_fn': None,
                'terminal_fn': None,

                'her_strategy':{
                    'resampling_probability': 0., # tune.grid_search([.5, 0.8]),
                    'type': 'future',
                }
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
                'store_last_n_paths': 20,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency':  DEFAULT_NUM_EPOCHS // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec


def get_variant_spec_classifier(universe,
                                domain,
                                task,
                                task_evaluation,
                                policy,
                                algorithm,
                                *args,
                                **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, task_evaluation, policy, algorithm, *args, **kwargs)

    classifier_layer_size = L = 256
    variant_spec['reward_classifier_params'] = {
        'type': 'feedforward_classifier',
        'kwargs': {
            'hidden_layer_sizes': (L, L),
            }
        }
    return variant_spec


def get_variant_spec(args):
    universe, domain = args.universe, args.domain
    # task, algorithm, n_epochs = args.task, args.algorithm, args.n_epochs
    # task = args.task = 'Image48SawyerPushNIPSEasyXY'
    task = args.task
    task_evaluation = args.task_evaluation

    algorithm, n_epochs = args.algorithm, args.n_epochs
    # active_query_frequency = args.active_query_frequency

    if args.algorithm in ['VICEGoalConditioned', 'VICEGANGoalConditioned']:
        variant_spec = get_variant_spec_classifier(
            universe, domain, task, task_evaluation, args.policy, args.algorithm)
    else:
        variant_spec = get_variant_spec_base(
            universe, domain, task, task_evaluation, args.policy, args.algorithm)

    binary_reward_tasks = [
            'Image48SawyerDoorHookMultiGoalResetFreeEnv-v0',
            'Image48SawyerDoorHookMultiGoalEnv-v0']
    distance_reward_tasks = [
            'Image48SawyerPushMultiGoalEnv-v0',
            'Image48SawyerPushMultiGoalTwoSmallPuckEnv-v0',
            'Image48SawyerPushMultiGoalTwoSmallPuckEasyEnv-v0',
            'Image48SawyerPushMultiGoalTwoPuckEnv-v0',
            'Image48SawyerPushMultiGoalThreeSmallPuckEnv-v0',
            'Image48SawyerPushMultiGoalCurriculumEnv-v0']

    # if task in binary_reward_tasks:
    #     relabel_reward = 1.0
    # elif task in distance_reward_tasks:
    #     relabel_reward = 0.0
    # else:
    #     raise NotImplementedError

    # variant_spec['replay_pool_params']['kwargs']['relabel_reward'] = relabel_reward

    # if args.algorithm in ['RAQ', 'VICERAQ']:
    #     variant_spec['algorithm_params']['kwargs']['active_query_frequency'] = \
    #         active_query_frequency

    variant_spec['algorithm_params']['kwargs']['n_epochs'] = n_epochs

    if 'Image' in task or 'Image48' in task:
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'conv_filters': (64,) * 3,
                'conv_kernel_sizes': (3,) * 3,
                'conv_strides': (2,) * 3,
                'normalization_type': 'layer',
                'downsampling_type': 'conv'
            },
        }

        variant_spec['policy_params']['kwargs'][
            'observation_preprocessors_params'] = {
                'pixels': deepcopy(preprocessor_params)
            }
        variant_spec['Q_params']['kwargs'][
            'observation_preprocessors_params'] = (
                tune.sample_from(lambda spec: (deepcopy(
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    ['observation_preprocessors_params']
                )))
            )
        # variant_spec['replay_pool_params']['kwargs']['max_size'] = (
        #     int(n_epochs * 1000))

        if args.algorithm in ('VICEGoalConditioned', 'VICEGANGoalConditioned'):
            variant_spec['reward_classifier_params']['kwargs'][
                'observation_preprocessors_params'] = (
                    tune.sample_from(lambda spec: (deepcopy(
                        spec.get('config', spec)
                        ['policy_params']
                        ['kwargs']
                        ['observation_preprocessors_params']
                    )))
                )

    elif 'Image' in task:
        raise NotImplementedError(
            "Add convnet preprocessor for this image input.")

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
