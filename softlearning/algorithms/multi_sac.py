import os
import uuid
from collections import OrderedDict
from numbers import Number

import skimage
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flatten_dict import flatten

from softlearning.models.utils import flatten_input_structure
from .rl_algorithm import RLAlgorithm
from softlearning.replay_pools.prioritized_experience_replay_pool import PrioritizedExperienceReplayPool

tfd = tfp.distributions


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class MultiSAC(RLAlgorithm):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policies,
            Qs,
            pools,
            goals=(),
            plotter=None,

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            her_iters=0,
            goal_classifier_params_directory=None,
            save_full_state=False,
            save_eval_paths=False,
            per_alpha=1,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(SAC, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policies = policies

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._goals = goals
        assert len(self._policies) == len(self._goals)
        assert len(self._Qs) == len(self._goals)

        self._pool = pool
        if isinstance(self._pool, PrioritizedExperienceReplayPool) and \
           self._pool._mode == 'Bellman_Error':
            self._per = True
            self._per_alpha = per_alpha
        else:
            self._per = False

        self._plotter = plotter

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize

        self._her_iters = her_iters
        self._base_env = training_environment.unwrapped

        self._save_full_state = save_full_state
        self._save_eval_paths = save_eval_paths
        self._goal_classifier_params_directory = goal_classifier_params_directory
        self._build()

    def _build(self):
        super(SAC, self)._build()
        if self._goal_classifier_params_directory:
            self._load_goal_classifier(self._goal_classifier_params_directory)
        else:
            self._goal_classifier = None

        self._init_actor_update()
        self._init_critic_update()
        self._init_diagnostics_ops()

    def _load_goal_classifier(self, goal_classifier_params_directory):
        import sys
        from goal_classifier.conv import CNN
        from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

        # print_tensors_in_checkpoint_file(goal_classifier_params_directory, all_tensors=False, tensor_name='')
        self._goal_classifier = CNN(goal_cond=True)
        variables = self._goal_classifier.get_variables()
        cnn_vars = [v for v in tf.trainable_variables() if v.name.split('/')[0] == 'goal_classifier']
        saver = tf.train.Saver(cnn_vars)
        saver.restore(self._session, goal_classifier_params_directory)

    def _classify_as_goals(self, images, goals):
        # NOTE: we can choose any goals we want.
        # Things to try:
        # - random goal
        # - single goal
        feed_dict = {self._goal_classifier.images: images,
            self._goal_classifier.goals: goals}
        # lid_pos = observations[:, -2]
        goal_probs = self._session.run(self._goal_classifier.pred_probs, feed_dict=feed_dict)[:, 1].reshape((-1, 1))
        return goal_probs

    def _get_Q_targets(self):
        Q_targets = []
        for i, policy in enumerate(self._policies):
            policy_inputs = flatten_input_structure({
                name: self._placeholders[i]['next_observations'][name]
            for name in policy.observation_keys
            })
            next_actions = policy.actions(policy_inputs)
            next_log_pis = policy.log_pis(policy_inputs, next_actions)

            next_Q_observations = {
                name: self._placeholders[i]['next_observations'][name]
                for name in self._Qs[i][0].observation_keys
            }
            next_Q_inputs = flatten_input_structure(
                {**next_Q_observations, 'actions': next_actions})
            next_Qs_values = tuple(Q(next_Q_inputs) for Q in self._Q_targets[i])

            min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
            next_values = min_next_Q - self._alpha * next_log_pis

            terminals = tf.cast(self._placeholders[i]['terminals'], next_values.dtype)

            Q_target = td_target(
                reward=self._reward_scale * self._placeholders[i]['rewards'],
                discount=self._discount,
                next_value=(1 - terminals) * next_values)
            Q_targets.append(tf.stop_gradient(Q_target))
        return Q_targets

    def _init_critic_updates(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_targets = self._get_Q_targets()
        assert len(Q_targets) == len(self._policies)
        for Q_target in Q_targets:
            assert Q_target.shape.as_list() == [None, 1]

        self._Q_optimizers = []
        self._Q_values = []
        self._Q_losses = []

        for i, Qs in enumerate(self._Qs):
            Q_observations = {
                name: self._placeholders[i]['observations'][name]
                for name in Qs[0].observation_keys
            }
            Q_inputs = flatten_input_structure({
                **Q_observations, 'actions': self._placeholders[i]['actions']})

            Q_values = tuple(Q(Q_inputs) for Q in Qs)
            self._Q_values.append(Q_values)

            Q_losses = tuple(
                tf.compat.v1.losses.mean_squared_error(
                    labels=Q_target, predictions=Q_value, weights=0.5)
                for Q_value in Q_values)
            self._Q_losses.append(Q_losses)

            self._bellman_errors.append(tf.reduce_min(tuple(
                tf.math.squared_difference(Q_target, Q_value)
                for Q_value in Q_values), axis=0))

            self._Q_optimizers.append(tuple(
                tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self._Q_lr,
                    name='{}_{}_{}_optimizer'.format(i, Q._name, j)
                ) for j, Q in enumerate(Qs)))

            Q_training_ops = tuple(
                Q_optimizer.minimize(loss=Q_loss, var_list=Q.trainable_variables)
                for i, (Q, Q_loss, Q_optimizer)
                in enumerate(zip(Qs, Q_losses, self._Q_optimizers[i])))

            self._training_ops.update({'Q_{}'.formate(i): tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """

        self._log_alphas = []
        self._alpha_optimizers = []
        self._alpha_train_ops = []
        self._alphas = []

        for i, policy in enumerate(self._policies):
            policy_inputs = flatten_input_structure({
                name: self._placeholders[i]['observations'][name]
                for name in policy.observation_keys
            })
            actions = policy.actions(policy_inputs)
            log_pis = policy.log_pis(policy_inputs, actions)

            assert log_pis.shape.as_list() == [None, 1]

            log_alpha = tf.compat.v1.get_variable(
                'log_alpha',
                dtype=tf.float32,
                initializer=0.0)
            alpha = tf.exp(log_alpha)
            self._log_alphas.append(log_alpha)
            self._alphas.append(alpha)

            if isinstance(self._target_entropy, Number):
                alpha_loss = -tf.reduce_mean(
                    log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

                self._alpha_optimizers.append(tf.compat.v1.train.AdamOptimizer(
                    self._policy_lr, name='alpha_optimizer_{i}'))
                self._alpha_train_ops.append(self._alpha_optimizers[i].minimize(
                    loss=alpha_loss, var_list=[log_alpha]))

                self._training_ops.update({
                    'temperature_alpha_{i}': self._alpha_train_ops[i]
                })

            if self._action_prior == 'normal':
                policy_prior = tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(self._action_shape),
                    scale_diag=tf.ones(self._action_shape))
                policy_prior_log_probs = policy_prior.log_prob(actions)
            elif self._action_prior == 'uniform':
                policy_prior_log_probs = 0.0

            Q_observations = {
                name: self._placeholders[i]['observations'][name]
                for name in self._Qs[i][0].observation_keys
            }
            Q_inputs = flatten_input_structure({
                **Q_observations, 'actions': actions})
            Q_log_targets = tuple(Q(Q_inputs) for Q in self._Qs[i])
            min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

            if self._reparameterize:
                policy_kl_losses = (
                    alpha * log_pis
                    - min_Q_log_target
                    - policy_prior_log_probs)
            else:
                raise NotImplementedError

            assert policy_kl_losses.shape.as_list() == [None, 1]

            self._policy_losses.append(policy_kl_losses)
            policy_loss = tf.reduce_mean(policy_kl_losses)

            self._policy_optimizers.append(tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._policy_lr,
                name="policy_optimizer_{i}"))

            policy_train_op = self._policy_optimizers[i].minimize(
                loss=policy_loss,
                var_list=policy.trainable_variables)

            self._training_ops.update({'policy_train_op_{i}': policy_train_op})

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value_{i}', self._Q_values[i]),
            ('Q_loss_{i}', self._Q_losses[i]),
            ('policy_loss_{i}', self._policy_losses[i]),
            ('alpha_{i}', self._alpha[i])
            for i in range(len(self._Qs))
        ))

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
        ))

        self._diagnostics_ops = OrderedDict([
            (f'{key}-{metric_name}', metric_fn(values))
            for key, values in diagnosables.items()
            for metric_name, metric_fn in diagnostic_metrics.items()
        ])

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Qs, Q_targets in zip(self._Qs, self._Q_targets):
            for Q, Q_target in zip(Qs, Q_targets):
                source_params = Q.get_weights()
                target_params = Q_target.get_weights()
                Q_target.set_weights([
                    tau * source + (1.0 - tau) * target
                    for source, target in zip(source_params, target_params)
                ])

    def _split_batch_by_goals(self, batch):
        """Split a batch into batches by goal"""
        batches = []
        for goal in self._goals:
            goal_inds = (batch['observations']['goal'] == goal)
            subbatch = {k: v[goal_inds] for k, v in batch.items()}
            batches.append(subbatch)
        return batches

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        batches = self._split_batch_by_goals(batch)
        feed_dict = self._get_feed_dict(iteration, batch)
        self._session.run(self._training_ops, feed_dict)

        # if self._her_iters:
        #     # Q: Is it better to build a large batch and take one grad step, or
        #     # resample many mini batches and take many grad steps?
        #     new_batches = {}
        #     for _ in range(self._her_iters):
        #         new_batch = self._get_goal_resamp_batch(batch)
        #         new_feed_dict = self._get_feed_dict(iteration, new_batch)
        #         self._session.run(self._training_ops, new_feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def get_bellman_error(self, batch):
        feed_dict = self._get_feed_dict(None, batch)

        ## TO TRY: weight by bellman error without entropy
        ## - sweep over per_alpha

        ## Question: why the min over the Q's?
        return self._session.run(self._bellman_errors, feed_dict)

    def _get_feed_dict(self, iteration, batches):
        """Construct a TensorFlow feed dictionary from multiple sample batches
        (one per policy)."""

        # TODO: funnel batch to placeholders by goal


        if np.random.rand() < 1e-3 and 'pixels' in batch['observations']:
            import os
            from skimage import io

            random_idx = np.random.randint(
                batch['observations']['pixels'].shape[0])
            image_save_dir = os.path.join(os.getcwd(), 'pixels')
            image_save_path = os.path.join(
                image_save_dir, f'observation_{iteration}_batch.png')
            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            io.imsave(image_save_path,
                      batch['observations']['pixels'][random_idx].copy())

        feed_dict = {}
        for i, batch in enumerate(batches):
            batch_flat = flatten(batch)
            placeholders_flat = flatten(self._placeholders[i])

            feed_dict.update({
                placeholders_flat[key]: batch_flat[key]
                for key in placeholders_flat.keys()
                if key in batch_flat.keys()
            })
            # if self._goal_classifier:
            #     if 'images' in batch.keys():
            #         images = batch['images']
            #         goal_sin = batch['observations'][:, -2].reshape((-1, 1))
            #         goal_cos = batch['observations'][:, -1].reshape((-1, 1))
            #         goals = np.arctan2(goal_sin, goal_cos)
            #     else:
            #         images = batch['observations'][:, :32*32*3].reshape((-1, 32, 32, 3))
            #     feed_dict[self._placeholders['rewards']] = self._classify_as_goals(images, goals)
            # else:
            feed_dict[self._placeholders[i]['rewards']] = batch['rewards']

            if iteration is not None:
                feed_dict[self._placeholders[i]['iteration']] = iteration

        return feed_dict

    # def _get_goal_resamp_batch(self, batch):
    #     new_goal = self._base_env.sample_goal()
    #     old_goal = self._base_env.get_goal()
    #     batch_obs = batch['observations']
    #     batch_act = batch['actions']
    #     batch_next_obs = batch['next_observations']

    #     new_batch_obs = self._base_env.relabel_obs_w_goal(batch_obs, new_goal)
    #     new_batch_next_obs = self._base_env.relabel_obs_w_goal(batch_next_obs, new_goal)

    #     if self._base_env.use_super_state_reward():
    #         batch_super_obs = batch['super_observations']
    #         new_batch_super_obs = super(self._base_env).relabel_obs_w_goal(batch_super_obs, new_goal)
    #         new_batch_rew = np.expand_dims(self._base_env.compute_rewards(new_batch_super_obs, batch_act)[0], 1)
    #     else:
    #         new_batch_rew = np.expand_dims(self._base_env.compute_rewards(new_batch_obs, batch_act)[0], 1)

    #     new_batch = {
    #         'rewards': new_batch_rew,
    #         'observations': new_batch_obs,
    #         'actions': batch['actions'],
    #         'next_observations': new_batch_next_obs,
    #         'terminals': batch['terminals'],
    #     }
    #     # (TODO) Implement relabeling of terminal flags
    #     return new_batch

    def _init_diagnostics_ops(self):
        self._diagnostics_ops = {
            **{
                f'{key}-{metric_name}': metric_fn(values)
                for key, values in (
                        ('Q_values', self._Q_values),
                        ('Q_losses', self._Q_losses),
                        ('policy_losses', self._policy_losses))
                for metric_name, metric_fn in (
                        ('mean', tf.reduce_mean),
                        ('std', lambda x: tfp.stats.stddev(
                            x, sample_axis=None)))
            },
            'alpha': self._alpha,
            'global_step': self.global_step,
        }

        return self._diagnostics_ops

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        batches = self._split_batch_by_goals(batch)
        feed_dict = self._get_feed_dict(iteration, batches)
        # TODO(hartikainen): We need to unwrap self._diagnostics_ops from its
        # tensorflow `_DictWrapper`.
        diagnostics = self._session.run({**self._diagnostics_ops}, feed_dict)

        diagnostics.update(OrderedDict([
            (f'policy/{key}', value)
            for key, value in
            self._policy.get_diagnostics(flatten_input_structure({
                name: batch['observations'][name]
                for name in self._policy.observation_keys
            })).items()
        ]))

        if self._goal_classifier:
            diagnostics.update({'goal_classifier/avg_reward': np.mean(feed_dict[self._rewards_ph])})

        if self._save_eval_paths:
            import pickle
            file_name = f'eval_paths_{iteration // self.epoch_length}.pkl'
            with open(os.path.join(os.getcwd(), file_name)) as f:
                pickle.dump(evaluation_paths, f)

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables