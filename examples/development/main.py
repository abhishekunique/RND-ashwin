
import os
import copy
import glob
import pickle
import sys
import numpy as np

import tensorflow as tf
from ray import tune

from softlearning.environments.utils import get_environment_from_params
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import (
    get_policy_from_variant, get_policy_from_params)
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant

from softlearning.misc.utils import set_seed, initialize_tf_variables
from examples.instrument import run_example_local

from softlearning.replay_pools.prioritized_experience_replay_pool import PrioritizedExperienceReplayPool
from softlearning.samplers.nn_sampler import NNSampler

tf.compat.v1.disable_eager_execution()


class ExperimentRunner(tune.Trainable):
    def _setup(self, variant):
        set_seed(variant['run_params']['seed'])

        self._variant = variant

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self._session = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        tf.keras.backend.set_session(self._session)

        self.train_generator = None
        self._built = False

    def _stop(self):
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()

    def _build(self):
        variant = copy.deepcopy(self._variant)
        environment_params = variant['environment_params']
        training_environment = self.training_environment = (
            get_environment_from_params(environment_params['training']))
        evaluation_environment = self.evaluation_environment = (
            get_environment_from_params(environment_params['evaluation'])
            if 'evaluation' in environment_params
            else training_environment)

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(variant, training_environment))
        Qs = self.Qs = get_Q_function_from_variant(
            variant, training_environment)
        Q_targets = self.Q_targets = get_Q_function_from_variant(
            variant, training_environment)

        policy = self.policy = get_policy_from_variant(
            variant, training_environment)

        try:
            if self.policy.preprocessors['pixels'].name == 'state_estimator_preprocessor':
                state_estimator = self.policy.preprocessors['pixels']

                from softlearning.replay_pools.flexible_replay_pool import Field
                replay_pool = self.replay_pool = (
                    get_replay_pool_from_variant(variant, training_environment,
                        extra_obs_keys_and_fields={
                            'object_state_prediction': Field(
                                name='object_state_prediction',
                                dtype=np.float32,
                                shape=(4,)
                            )
                        }))
            else:
                state_estimator = None
        except:
            state_estimator = None

        sampler = self.sampler = get_sampler_from_variant(
            variant,
            state_estimator=state_estimator)

        last_checkpoint_dir = variant['replay_pool_params']['last_checkpoint_dir']
        if last_checkpoint_dir:
            print('restoring')
            self._restore_replay_pool(last_checkpoint_dir)

        if isinstance(sampler, NNSampler):
            print('restoring nn_pool')
            nn_pool_dir = variant['sampler_params']['nn_pool_dir']
            nn_pool = (get_replay_pool_from_variant(variant, training_environment))

            replay_pool = self.replay_pool
            self.replay_pool = nn_pool
            self._restore_replay_pool(nn_pool_dir)
            self.replay_pool = replay_pool
            self.sampler.initialize_nn_pool(nn_pool)

        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy_from_params(
                variant['exploration_policy_params'], training_environment))

        # VAE
        if self.policy.preprocessors['pixels'].name == 'vae_preprocessor':
            from softlearning.models.utils import get_vae
            vae = get_vae(**variant['policy_params']
                                   ['kwargs']
                                   ['observation_preprocessors_params']
                                   ['pixels']
                                   ['kwargs'])
        else:
            vae = None

        if variant['algorithm_params']['rnd_params']:
            from softlearning.rnd.utils import get_rnd_networks_from_variant
            rnd_networks = get_rnd_networks_from_variant(variant, training_environment)
        else:
            rnd_networks = ()

        self.algorithm = get_algorithm_from_variant(
            variant=self._variant,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            Q_targets=Q_targets,
            pool=replay_pool,
            sampler=sampler,
            session=self._session,
            state_estimator=state_estimator,
            vae=vae,
            rnd_networks=rnd_networks)

        if isinstance(replay_pool, PrioritizedExperienceReplayPool) and \
           replay_pool._mode == 'Bellman_Error':
            replay_pool.initialize(self.algorithm)

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True

    def _train(self):
        if not self._built:
            self._build()

        if self.train_generator is None:
            self.train_generator = self.algorithm.train()

        diagnostics = next(self.train_generator)

        return diagnostics

    def _pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint.pkl')

    def _replay_pool_pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    def _tf_checkpoint_prefix(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint')

    def _get_tf_checkpoint(self):
        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)

        return tf_checkpoint

    @property
    def picklables(self):
        return {
            'variant': self._variant,
            'training_environment': self.training_environment,
            'evaluation_environment': self.evaluation_environment,
            'sampler': self.sampler,
            'algorithm': self.algorithm,
            'policy_weights': self.policy.get_weights(),
        }

    def _save_value_functions(self, checkpoint_dir):
        if isinstance(self.Qs, tf.keras.Model):
            Qs = [self.Qs]
        elif isinstance(self.Qs, (list, tuple)):
            Qs = self.Qs
        else:
            raise TypeError(self.Qs)

        for i, Q in enumerate(Qs):
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'Qs_{i}')
            Q.save_weights(checkpoint_path)

    def _restore_value_functions(self, checkpoint_dir):
        if isinstance(self.Qs, tf.keras.Model):
            Qs = [self.Qs]
        elif isinstance(self.Qs, (list, tuple)):
            Qs = self.Qs
        else:
            raise TypeError(self.Qs)

        for i, Q in enumerate(Qs):
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'Qs_{i}')
            Q.load_weights(checkpoint_path)

    def _save(self, checkpoint_dir):
        """Implements the checkpoint logic.

        TODO(hartikainen): This implementation is currently very hacky. Things
        that need to be fixed:
          - Figure out how serialize/save tf.keras.Model subclassing. The
            current implementation just dumps the weights in a pickle, which
            is not optimal.
          - Try to unify all the saving and loading into easily
            extendable/maintainable interfaces. Currently we use
            `tf.train.Checkpoint` and `pickle.dump` in very unorganized way
            which makes things not so usable.
        """
        pickle_path = self._pickle_path(checkpoint_dir)
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.picklables, f)

        self._save_value_functions(checkpoint_dir)

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._save_replay_pool(checkpoint_dir)

        tf_checkpoint = self._get_tf_checkpoint()

        tf_checkpoint.save(
            file_prefix=self._tf_checkpoint_prefix(checkpoint_dir),
            session=self._session)

        return os.path.join(checkpoint_dir, '')

    def _save_replay_pool(self, checkpoint_dir):
        replay_pool_pickle_path = self._replay_pool_pickle_path(
            checkpoint_dir)
        self.replay_pool.save_latest_experience(replay_pool_pickle_path)

    def _restore_replay_pool(self, current_checkpoint_dir):
        # experiment_root = os.path.dirname(current_checkpoint_dir)
        experiment_root = current_checkpoint_dir
        experience_paths = [
            self._replay_pool_pickle_path(checkpoint_dir)
            for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_root, 'checkpoint_*')))
        ]

        for experience_path in experience_paths:
            self.replay_pool.load_experience(experience_path)

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
        Qs = self.Qs = get_Q_function_from_variant(
            self._variant, training_environment)
        self._restore_value_functions(checkpoint_dir)
        policy = self.policy = (
            get_policy_from_variant(self._variant, training_environment))
        self.policy.set_weights(picklable['policy_weights'])
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy_from_params(
                self._variant['exploration_policy_params'],
                training_environment))

        self.algorithm = get_algorithm_from_variant(
            variant=self._variant,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            sampler=sampler,
            session=self._session)
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


def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    run_example_local('examples.development', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
