import numpy as np
import tensorflow as tf

from .sac import SAC
from softlearning.models.utils import flatten_input_structure
from flatten_dict import flatten


class DDL(SAC):
    def __init__(
        self,
        distance_fn,
        goal_state,
        train_distance_fn_every_n_steps=16,
        ddl_lr=3e-4,
        ddl_batch_size=256,
        **kwargs,
    ):
        # TODO: Make a goal proposer
        self._distance_fn = distance_fn
        self._goal_state = goal_state

        self._train_distance_fn_every_n_steps = train_distance_fn_every_n_steps
        self._ddl_lr = ddl_lr
        self._ddl_batch_size = ddl_batch_size

        super(DDL, self).__init__(**kwargs)

    def _build(self):
        super(DDL, self)._build()
        self._init_ddl_update()

    def _init_placeholders(self):
        super(DDL, self)._init_placeholders()
        self._init_ddl_placeholders()

    def _init_ddl_placeholders(self):
        self._placeholders.update({
            's1': {
                name: tf.compat.v1.placeholder(
                    dtype=(
                        np.float32
                        if np.issubdtype(observation_space.dtype, np.floating)
                        else observation_space.dtype
                    ),
                    shape=(None, *observation_space.shape),
                    name=f's1/{name}')
                for name, observation_space
                in self._training_environment.observation_space.spaces.items()
            },
            's2': {
                name: tf.compat.v1.placeholder(
                    dtype=(
                        np.float32
                        if np.issubdtype(observation_space.dtype, np.floating)
                        else observation_space.dtype
                    ),
                    shape=(None, *observation_space.shape),
                    name=f's2/{name}')
                for name, observation_space
                in self._training_environment.observation_space.spaces.items()
            },
            'distances': tf.compat.v1.placeholder(
                dtype=np.float32, shape=(None, 1), name='distances')
        })

    def _init_ddl_update(self):
        distance_fn_inputs = self._distance_fn_inputs(
            s1=self._placeholders['s1'], s2=self._placeholders['s2'])
        distance_preds = self._distance_preds = (
            self._distance_fn(distance_fn_inputs))

        distance_targets = self._placeholders['distances']

        distance_loss = self._distance_loss = (
            tf.compat.v1.losses.mean_squared_error(
                labels=distance_targets,
                predictions=distance_preds,
                weights=0.5)
        )

        ddl_optimizer = self._ddl_optimizer = (
            tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._ddl_lr,
                name='ddl_optimizer'))
        self._ddl_train_op = ddl_optimizer.minimize(
            loss=distance_loss,
            var_list=self._distance_fn.trainable_variables)

    def _get_ddl_feed_dict(self):
        s1_indices = self.sampler.pool.random_indices(self._ddl_batch_size)
        max_path_length = self.sampler.max_path_length

        distances = np.random.randint(max_path_length - s1_indices % max_path_length)
        s2_indices = s1_indices + distances

        s1 = self.sampler.pool.batch_by_indices(s1_indices)
        s2 = self.sampler.pool.batch_by_indices(s2_indices)
        distances = distances.astype(np.float32)[..., None]
        feed_dict = {
            **{
                self._placeholders['s1'][key]: s1['observations'][key]
                for key in self._distance_fn.observation_keys
            },
            **{
                self._placeholders['s2'][key]: s2['observations'][key]
                for key in self._distance_fn.observation_keys
            },
            self._placeholders['distances']: distances,
        }
        return feed_dict

    def _distance_fn_inputs(self, s1, s2):
        inputs_1 = {
            name: s1[name]
            for name in self._distance_fn.observation_keys
        }
        inputs_2 = {
            name: s2[name]
            for name in self._distance_fn.observation_keys
        }
        inputs = {
            's1': inputs_1,
            's2': inputs_2,
        }
        return flatten_input_structure(inputs)

    def _policy_inputs(self, observations):
        policy_inputs = flatten_input_structure({
            name: observations[name]
            for name in self._policy.observation_keys
        })
        return policy_inputs

    def _Q_inputs(self, observations, actions):
        Q_observations = {
            name: observations[name]
            for name in self._Qs[0].observation_keys
        }
        Q_inputs = flatten_input_structure(
            {**Q_observations, 'actions': actions})
        return Q_inputs

    def _init_extrinsic_reward(self):
        """
        Initializes the DDL reward as -(distance to goal)
        The feed dict should set one of the s1/s2 placeholders the goal
        """
        distance_fn_inputs = self._distance_fn_inputs(
            s1=self._placeholders['s1'], s2=self._placeholders['s2'])
        distances_to_goal = self._distance_fn(distance_fn_inputs)
        self._unscaled_ext_reward = -distances_to_goal

    def _get_feed_dict(self, iteration, batch):
        feed_dict = super(DDL, self)._get_feed_dict(iteration, batch)
        placeholders_flat = flatten(self._placeholders)

        # === Set s1, s2 for training Qs ===
        feed_dict.update({
            self._placeholders['s1'][key]: feed_dict[self._placeholders['observations'][key]]
            for key in self._distance_fn.observation_keys
        })

        batch_size = feed_dict[next(iter(feed_dict))].shape[0]
        feed_dict.update({
            self._placeholders['s2'][key]:
            np.repeat(self._goal_state[key][None], batch_size, axis=0)
            for key in self._distance_fn.observation_keys
        })

        return feed_dict

    def _do_training(self, iteration, batch):
        super(DDL, self)._do_training(iteration, batch)
        if iteration % self._train_distance_fn_every_n_steps == 0:
            ddl_feed_dict = self._get_ddl_feed_dict()
            self._session.run(self._ddl_train_op, ddl_feed_dict)

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(DDL, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        ddl_feed_dict = self._get_ddl_feed_dict()
        goal_feed_dict = self._get_feed_dict(iteration, batch)

        overall_distance_loss = self._session.run(
            self._distance_loss,
            feed_dict=ddl_feed_dict)

        goal_relative_distance_preds = self._session.run(
            self._distance_preds,
            feed_dict=goal_feed_dict)

        diagnostics.update({
            'ddl/overall_distance_loss': np.mean(overall_distance_loss),
            'ddl/goal_relative_distance_preds': np.mean(goal_relative_distance_preds),
        })

        return diagnostics
