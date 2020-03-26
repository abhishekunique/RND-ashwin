import sys
from softlearning.policies.utils import (
    get_policy_from_variant, get_policy_from_params, get_policy)
from softlearning.models.utils import get_distance_estimator_from_variant
from softlearning.misc.generate_goal_examples import get_ddl_goal_state_from_variant

from examples.instrument import run_example_local
from examples.development.main import ExperimentRunner


class ExperimentRunnerDistanceLearning(ExperimentRunner):
    def _get_algorithm_kwargs(self, variant):
        algorithm_kwargs = super()._get_algorithm_kwargs(variant)
        algorithm_type = variant['algorithm_params']['type']

        if algorithm_type == 'DDL':
            env = algorithm_kwargs['training_environment']
            distance_fn = self.distance_fn = (
                get_distance_estimator_from_variant(
                    self._variant, env))
            algorithm_kwargs['distance_fn'] = distance_fn
            algorithm_kwargs['goal_state'] = (
                get_ddl_goal_state_from_variant(self._variant))

        return algorithm_kwargs

    def _restore_algorithm_kwargs(self, picklable, checkpoint_dir, variant):
        algorithm_kwargs = super()._restore_algorithm_kwargs(picklable, checkpoint_dir, variant)

        if 'distance_estimator' in picklable.keys():
            distance_fn = self.distance_fn = picklable['distance_estimator']
            algorithm_kwargs['distance_fn'] = reward_classifier

        return algorithm_kwargs

    # def _restore_multi_algorithm_kwargs(self, picklable, checkpoint_dir, variant):
    #     algorithm_kwargs = super()._restore_multi_algorithm_kwargs(
    #         picklable, checkpoint_dir, variant)

    #     if 'reward_classifiers' in picklable.keys():

    #         reward_classifiers = self.reward_classifiers = picklable[
    #             'reward_classifiers']
    #         for reward_classifier in self.reward_classifiers:
    #             reward_classifier.observation_keys = (variant['reward_classifier_params']
    #                                                          ['kwargs']
    #                                                          ['observation_keys'])

    #         algorithm_kwargs['classifiers'] = reward_classifiers
    #         goal_pools_train, goal_pools_validation = (
    #             get_example_pools_from_variant(variant))
    #         algorithm_kwargs['goal_example_pools'] = goal_pools_train
    #         algorithm_kwargs['goal_example_validation_pools'] = goal_pools_validation
    #     return algorithm_kwargs

    @property
    def picklables(self):
        picklables = super().picklables

        picklables['distance_estimator'] = self.distance_fn

        return picklables


def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    # __package__ should be `distance_learning.main`
    run_example_local('examples.distance_learning', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
