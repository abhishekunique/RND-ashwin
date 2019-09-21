import os
import copy
import pickle
import sys

import tensorflow as tf

from softlearning.environments.utils import (
    get_goal_example_environment_from_variant, get_environment_from_params)
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import (
    get_policy_from_variant, get_policy_from_params, get_policy)
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.models.utils import get_reward_classifier_from_variant
from softlearning.misc.generate_goal_examples import (
    get_goal_example_from_variant)
from softlearning.misc.get_multigoal_example_pools import (
    get_example_pools_from_variant)
from softlearning.misc.utils import initialize_tf_variables
from examples.instrument import run_example_local
from examples.development.main import ExperimentRunner
from softlearning.environments.adapters.gym_adapter import GymAdapter


class ExperimentRunnerClassifierRL(ExperimentRunner):
    def _get_algorithm_kwargs(self, variant):
        algorithm_kwargs = super()._get_algorithm_kwargs(variant)
        # === LOAD SINGLE GOAL POOL ===
        if variant['algorithm_params']['type'] in ['SACClassifier', 'RAQ', 'VICE', 'VICEGAN', 'VICERAQ']:
            reward_classifier = self.reward_classifier = (
                get_reward_classifier_from_variant(
                    self._variant, algorithm_kwargs['training_environment']))
            algorithm_kwargs['classifier'] = reward_classifier

            goal_examples_train, goal_examples_validation = (
                get_goal_example_from_variant(variant))
            algorithm_kwargs['goal_examples'] = goal_examples_train
            algorithm_kwargs['goal_examples_validation'] = (
                goal_examples_validation)

        # === LOAD GOAL POOLS FOR MULTI GOAL ===
        elif variant['algorithm_params']['type'] in ['VICEGANMultiGoal', 'MultiVICEGAN']:
            goal_pools_train, goal_pools_validation = (
                get_example_pools_from_variant(variant))
            num_goals = len(goal_pools_train)

            reward_classifiers = self._reward_classifiers = [get_reward_classifier_from_variant(
                variant, algorithm_kwargs['training_environment']) for _ in range(num_goals)]

            algorithm_kwargs['classifiers'] = reward_classifiers
            algorithm_kwargs['goal_example_pools'] = goal_pools_train
            algorithm_kwargs['goal_example_validation_pools'] = goal_pools_validation

        return algorithm_kwargs

    def _restore_algorithm_kwargs(self, checkpoint_dir):
        with self._session.as_default():
            pickle_path = self._pickle_path(checkpoint_dir)
            with open(pickle_path, 'rb') as f:
                picklable = pickle.load(f)
        algorithm_kwargs = super()._restore_algorithm_kwargs(checkpoint_dir)

        if 'reward_classifier' in picklable.keys():
            reward_classifier = self.reward_classifier = picklable[
                'reward_classifier']
            algorithm_kwargs['classifier'] = reward_classifier

            goal_examples_train, goal_examples_validation = (
                get_goal_example_from_variant(self._variant))
            algorithm_kwargs['goal_examples'] = goal_examples_train
            algorithm_kwargs['goal_examples_validation'] = (
                goal_examples_validation)

    def _restore_multi_algorithm_kwargs(self, checkpoint_dir):
        with self._session.as_default():
            pickle_path = self._pickle_path(checkpoint_dir)
            with open(pickle_path, 'rb') as f:
                picklable = pickle.load(f)
        algorithm_kwargs = super()._restore_algorithm_kwargs(checkpoint_dir)

        if 'reward_classifiers' in picklable.keys():
            reward_classifiers = self.reward_classifiers = picklable[
                'reward_classifiers']
            algorithm_kwargs['classifiers'] = reward_classifiers

            goal_pools_train, goal_pools_validation = (
                get_example_pools_from_variant(self._variant))
            algorithm_kwargs['classifiers'] = reward_classifiers
            algorithm_kwargs['goal_example_pools'] = goal_pools_train
            algorithm_kwargs['goal_example_validation_pools'] = goal_pools_validation

    @property
    def picklables(self):
        picklables = super().picklables

        if hasattr(self, 'reward_classifier'):
            picklables['reward_classifier'] = self.reward_classifier
        elif hasattr(self, 'reward_classifiers'):
            picklables['reward_classifiers'] = self.reward_classifiers

        return picklables


def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    # __package__ should be `development.main`
    run_example_local('examples.classifier_rl', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
