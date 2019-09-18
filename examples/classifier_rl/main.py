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

    def _build(self):
        variant = copy.deepcopy(self._variant)

        train_env_params = variant['environment_params']['training']
        eval_env_params = variant['environment_params']['evaluation']
        training_environment = self.training_environment = (
            get_environment_from_params(train_env_params))
        evaluation_environment = self.evaluation_environment = (
            get_environment_from_params(eval_env_params))

        # training_environment = self.training_environment = (
        #     GymAdapter(domain=domain, task=task, **variant['env_params']))
        # eval_params = variant['env_params'].copy()
        # eval_params['swap_goals_upon_completion'] = False
        # evaluation_environment = self.evaluation_environment = (
        #     GymAdapter(domain=domain, task=task_evaluation, **eval_params))

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(variant, training_environment))
        sampler = self.sampler = get_sampler_from_variant(variant)
        Qs = self.Qs = get_Q_function_from_variant(
            variant, training_environment)
        policy = self.policy = get_policy_from_variant(
            variant, training_environment)
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy_from_params(
                variant['exploration_policy_params'], training_environment))
        algorithm_kwargs = {
            'variant': variant,
            'training_environment': training_environment,
            'evaluation_environment': evaluation_environment,
            'policy': policy,
            'initial_exploration_policy': initial_exploration_policy,
            'Qs': Qs,
            'pool': replay_pool,
            'sampler': sampler,
            'session': self._session,
        }
        if self._variant['algorithm_params']['type'] in ['SACClassifier', 'RAQ', 'VICE', 'VICEGAN', 'VICERAQ']:
            reward_classifier = self.reward_classifier = (
                get_reward_classifier_from_variant(
                    self._variant, training_environment))
            algorithm_kwargs['classifier'] = reward_classifier

            goal_examples_train, goal_examples_validation = (
                get_goal_example_from_variant(variant))
            algorithm_kwargs['goal_examples'] = goal_examples_train
            algorithm_kwargs['goal_examples_validation'] = (
                goal_examples_validation)

        # TODO: Remove TwoGoal, generalize with the multigoal
        if self._variant['algorithm_params']['type'] in ['VICEGANTwoGoal']:
            reward_classifier_0 = self._reward_classifier_0 = (
                get_reward_classifier_from_variant(
                    self._variant, training_environment))
            reward_classifier_1 = self._reward_classifier_1 = (
                get_reward_classifier_from_variant(
                    self._variant, training_environment))
            algorithm_kwargs['classifier_0'] = reward_classifier_0
            algorithm_kwargs['classifier_1'] = reward_classifier_1

            goal_pools_train, goal_pools_validation = (
                get_example_pools_from_variant(variant))

            goal_examples_0, goal_examples_1 = goal_pools_train
            goal_examples_validation_0, goal_examples_validation_1 = goal_pools_validation
            algorithm_kwargs['goal_examples_0'] = goal_examples_0
            algorithm_kwargs['goal_examples_1'] = goal_examples_1
            algorithm_kwargs['goal_examples_validation_0'] = goal_examples_validation_0
            algorithm_kwargs['goal_examples_validation_1'] = goal_examples_validation_1

        elif self._variant['algorithm_params']['type'] in ['VICEGANMultiGoal']:
            goal_pools_train, goal_pools_validation = (
                get_example_pools_from_variant(variant))
            num_goals = len(goal_pools_train)

            reward_classifiers = [get_reward_classifier_from_variant(
                variant, training_environment) for _ in range(num_goals)]

            algorithm_kwargs['classifiers'] = reward_classifiers
            algorithm_kwargs['goal_example_pools'] = goal_pools_train
            algorithm_kwargs['goal_example_validation_pools'] = goal_pools_validation

        # RND
        if variant['algorithm_params']['rnd_params']:
            from softlearning.rnd.utils import get_rnd_networks_from_variant
            rnd_networks = get_rnd_networks_from_variant(variant, training_environment)
        else:
            rnd_networks = ()
        algorithm_kwargs['rnd_networks'] = rnd_networks

        self.algorithm = get_algorithm_from_variant(**algorithm_kwargs)

        # Give the algorithm to the sampler to calculate reward
        if variant['sampler_params']['type'] == 'ClassifierSampler':
            sampler.set_algorithm(self.algorithm)

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True

    def _restore(self, checkpoint_dir):
        assert isinstance(checkpoint_dir, str), checkpoint_dir

        checkpoint_dir = checkpoint_dir.rstrip('/')

        with self._session.as_default():
            pickle_path = self._pickle_path(checkpoint_dir)
            with open(pickle_path, 'rb') as f:
                picklable = pickle.load(f)

        training_environment = self.training_environment = picklable[
            'training_environment']
        evaluation_environment = self.evaluation_environment = picklable[
            'evaluation_environment']

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(self._variant, training_environment))

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._restore_replay_pool(checkpoint_dir)

        sampler = self.sampler = picklable['sampler']
        Qs = self.Qs = picklable['Qs']
        # policy = self.policy = picklable['policy']
        policy = self.policy = (
            get_policy_from_variant(self._variant, training_environment, Qs))
        self.policy.set_weights(picklable['policy_weights'])
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', training_environment))

        algorithm_kwargs = {
            'variant': self._variant,
            'training_environment': self.training_environment,
            'evaluation_environment': self.evaluation_environment,
            'policy': policy,
            'initial_exploration_policy': initial_exploration_policy,
            'Qs': Qs,
            'pool': replay_pool,
            'sampler': sampler,
            'session': self._session,
        }

        if self._variant['algorithm_params']['type'] in (
                'SACClassifier', 'RAQ', 'VICE', 'VICEGAN', 'VICERAQ'):
            reward_classifier = self.reward_classifier = picklable[
                'reward_classifier']
            algorithm_kwargs['classifier'] = reward_classifier

            goal_examples_train, goal_examples_validation = (
                get_goal_example_from_variant(self._variant))
            algorithm_kwargs['goal_examples'] = goal_examples_train
            algorithm_kwargs['goal_examples_validation'] = (
                goal_examples_validation)

        self.algorithm = get_algorithm_from_variant(**algorithm_kwargs)
        self.algorithm.__setstate__(picklable['algorithm'].__getstate__())

        tf_checkpoint = self._get_tf_checkpoint()
        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            os.path.split(self._tf_checkpoint_prefix(checkpoint_dir))[0]))

        status.assert_consumed().run_restore_ops(self._session)
        initialize_tf_variables(self._session, only_uninitialized=True)

        # TODO(hartikainen): target Qs should either be checkpointed or pickled.
        for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
            Q_target.set_weights(Q.get_weights())

        self._built = True

    @property
    def picklables(self):
        picklables = {
            'variant': self._variant,
            'training_environment': self.training_environment,
            'evaluation_environment': self.evaluation_environment,
            'sampler': self.sampler,
            'algorithm': self.algorithm,
            'Qs': self.Qs,
            'policy_weights': self.policy.get_weights(),
        }

        if hasattr(self, 'reward_classifier'):
            picklables['reward_classifier'] = self.reward_classifier

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
