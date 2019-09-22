import os
from collections import OrderedDict
from numbers import Number

import skimage
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flatten_dict import flatten

from softlearning.models.utils import flatten_input_structure
from .rl_algorithm import RLAlgorithm
from softlearning.replay_pools.prioritized_experience_replay_pool import (
    PrioritizedExperienceReplayPool
)
from softlearning.misc.utils import angle_distance
from softlearning.models.state_estimation import normalize

tfd = tfp.distributions


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class SAC(RLAlgorithm):
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
            policy,
            Qs,
            pool,
            Q_targets=None,

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
            normalize_ext_reward_gamma=1,
            ext_reward_coeff=1,
            state_estimator=None,
            state_estimator_iters=None,
            train_state_estimator_online=False,

            vae=None,

            rnd_networks=(),
            rnd_lr=1e-4,
            rnd_int_rew_coeff=0,
            rnd_gamma=0.99,
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
            target_update_interval ('int', [grad_steps]): Frequency at which target network
                updates occur.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(SAC, self).__init__(**kwargs)
        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = (
            Q_targets if Q_targets
            else tuple(tf.keras.models.clone_model(Q) for Q in Qs)
        )

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

        # State estimator
        self._state_estimator = state_estimator
        self._state_estimator_iters = state_estimator_iters
        self._train_state_estimator_online = train_state_estimator_online
        self._state_estimator_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        # VAE
        self._vae = vae
        self._preprocessed_Q_inputs = self._Qs[0].preprocessed_inputs_fn

        self._normalize_ext_reward_gamma = normalize_ext_reward_gamma
        self._ext_reward_coeff = ext_reward_coeff
        self._running_ext_rew_std = 1
        self._rnd_int_rew_coeff = 0

        if rnd_networks:
            self._rnd_target = rnd_networks[0]
            self._rnd_predictor = rnd_networks[1]
            self._rnd_lr = rnd_lr
            self._rnd_int_rew_coeff = rnd_int_rew_coeff
            self._rnd_gamma = rnd_gamma
            self._running_int_rew_std = 1
            # self._rnd_gamma = 1
            # self._running_int_rew_std = 1
        self._build()

    def _build(self):
        super(SAC, self)._build()
        if self._goal_classifier_params_directory:
            self._load_goal_classifier(self._goal_classifier_params_directory)
        else:
            self._goal_classifier = None

        self._init_external_reward()
        if self._rnd_int_rew_coeff:
            self._init_rnd_update()
        self._init_actor_update()
        self._init_critic_update()
        self._init_diagnostics_ops()

    def _load_goal_classifier(self, goal_classifier_params_directory):
        from goal_classifier.conv import CNN

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
        feed_dict = {
            self._goal_classifier.images: images,
            self._goal_classifier.goals: goals
        }
        # lid_pos = observations[:, -2]
        goal_probs = self._session.run(self._goal_classifier.pred_probs, feed_dict=feed_dict)[:, 1].reshape((-1, 1))
        return goal_probs

    def _init_external_reward(self):
        self._unscaled_ext_reward = self._placeholders['rewards']

    def _get_Q_target(self):
        policy_inputs = flatten_input_structure({
            name: self._placeholders['next_observations'][name]
            for name in self._policy.observation_keys
        })
        next_actions = self._policy.actions(policy_inputs)
        next_log_pis = self._policy.log_pis(policy_inputs, next_actions)

        next_Q_observations = {
            name: self._placeholders['next_observations'][name]
            for name in self._Qs[0].observation_keys
        }
        next_Q_inputs = flatten_input_structure(
            {**next_Q_observations, 'actions': next_actions})
        next_Qs_values = tuple(Q(next_Q_inputs) for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_values = min_next_Q - self._alpha * next_log_pis

        terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

        if self._rnd_int_rew_coeff:
            self._unscaled_int_reward = tf.clip_by_value(
                self._rnd_errors / self._placeholders['reward']['running_int_rew_std'],
                0, 1000
            )
            self._int_reward = self._rnd_int_rew_coeff * self._unscaled_int_reward
        else:
            self._int_reward = 0
        self._normalized_ext_reward = self._unscaled_ext_reward / self._placeholders['reward']['running_ext_rew_std']
        self._ext_reward = self._normalized_ext_reward * self._ext_reward_coeff
        self._total_reward = self._ext_reward + self._int_reward

        Q_target = td_target(
            reward=self._reward_scale * self._total_reward,
            discount=self._discount,
            next_value=(1 - terminals) * next_values)
        return tf.stop_gradient(Q_target)

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_target = self._get_Q_target()
        assert Q_target.shape.as_list() == [None, 1]

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_inputs = flatten_input_structure({
            **Q_observations, 'actions': self._placeholders['actions']})
        Q_values = self._Q_values = tuple(Q(Q_inputs) for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.compat.v1.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._bellman_errors = tf.reduce_min(tuple(
            tf.math.squared_difference(Q_target, Q_value)
            for Q_value in Q_values), axis=0)

        self._Q_optimizers = tuple(
            tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))

        Q_training_ops = tuple(
            Q_optimizer.minimize(loss=Q_loss, var_list=Q.trainable_variables)
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        policy_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._policy.observation_keys
        })
        actions = self._policy.actions(policy_inputs)
        log_pis = self._policy.log_pis(policy_inputs, actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.compat.v1.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.compat.v1.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_inputs = flatten_input_structure({
            **Q_observations, 'actions': actions})
        Q_log_targets = tuple(Q(Q_inputs) for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - min_Q_log_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        self._policy_losses = policy_kl_losses
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        policy_train_op = self._policy_optimizer.minimize(
            loss=policy_loss,
            var_list=self._policy.trainable_variables)

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_rnd_update(self):
        self._placeholders['reward'].update({
            'running_int_rew_std': tf.compat.v1.placeholder(
                tf.float32, shape=(), name='running_int_rew_std')
        })
        policy_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._policy.observation_keys
        })

        targets = tf.stop_gradient(self._rnd_target(policy_inputs))
        predictions = self._rnd_predictor(policy_inputs)

        self._rnd_errors = tf.expand_dims(tf.reduce_mean(
            tf.math.squared_difference(targets, predictions), axis=-1), 1)
        self._rnd_loss = tf.reduce_mean(self._rnd_errors)
        self._rnd_error_std = tf.math.reduce_std(self._rnd_errors)
        self._rnd_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._rnd_lr,
            name="rnd_optimizer")
        rnd_train_op = self._rnd_optimizer.minimize(
            loss=self._rnd_loss)
        self._training_ops.update(
            {'rnd_train_op': rnd_train_op}
        )

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value', self._Q_values),
            ('Q_loss', self._Q_losses),
            ('policy_loss', self._policy_losses),
            ('alpha', self._alpha)
        ))

        if self._rnd_int_rew_coeff:
            diagnosables['rnd_reward'] = self._int_reward
            diagnosables['rnd_error'] = self._rnd_errors
            diagnosables['running_rnd_reward_std'] = self._placeholders[
                'reward']['running_int_rew_std']

        diagnosables['normalized_ext_reward'] = self._normalized_ext_reward
        diagnosables['ext_reward'] = self._ext_reward

        diagnosables['running_ext_reward_std'] = self._placeholders['reward']['running_ext_rew_std']
        diagnosables['total_reward'] = self._total_reward

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
            ('max', tf.math.reduce_max),
            ('min', tf.math.reduce_min),
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

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        feed_dict = self._get_feed_dict(iteration, batch)
        # intermediate = self._Qs[0].preprocessed_inputs_fn
        # inputs = (
        #     feed_dict[self._placeholders['actions']],
        #     feed_dict[self._placeholders['observations']['claw_qpos']],
        #     feed_dict[self._placeholders['observations']['last_action']],
        #     feed_dict[self._placeholders['observations']['pixels']],
        #     feed_dict[self._placeholders['observations']['target_xy_position']],
        #     feed_dict[self._placeholders['observations']['target_z_orientation_cos']],
        #     feed_dict[self._placeholders['observations']['target_z_orientation_sin']],
        # )
        # feed_dict_interm = {
        #     input_ph: input_np
        #     for input_ph, input_np in zip(intermediate.inputs, inputs)
        # }

        self._session.run(self._training_ops, feed_dict)
        if self._rnd_int_rew_coeff:
            int_rew_std = np.maximum(np.std(self._session.run(self._unscaled_int_reward, feed_dict)), 1e-3)
            self._running_int_rew_std = self._running_int_rew_std * self._rnd_gamma + int_rew_std * (1-self._rnd_gamma)

        if self._normalize_ext_reward_gamma != 1:
            ext_rew_std = np.maximum(np.std(self._session.run(self._normalized_ext_reward, feed_dict)), 1e-3)
            self._running_ext_rew_std = self._running_ext_rew_std * self._normalize_ext_reward_gamma + \
                ext_rew_std * (1-self._normalize_ext_reward_gamma)

        if self._her_iters:
            # Q: Is it better to build a large batch and take one grad step, or
            # resample many mini batches and take many grad steps?
            new_batches = {}
            for _ in range(self._her_iters):
                new_batch = self._get_goal_resamp_batch(batch)
                new_feed_dict = self._get_feed_dict(iteration, new_batch)
                self._session.run(self._training_ops, new_feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def get_bellman_error(self, batch):
        feed_dict = self._get_feed_dict(None, batch)

        ## TO TRY: weight by bellman error without entropy
        ## - sweep over per_alpha

        ## Question: why the min over the Q's?
        return self._session.run(self._bellman_errors, feed_dict)

    def _get_feed_dict(self, iteration, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        batch_flat = flatten(batch)
        placeholders_flat = flatten(self._placeholders)

        if np.random.rand() < 1e-4 and 'pixels' in batch['observations']:
            import os
            from skimage import io
            random_idx = np.random.randint(
                batch['observations']['pixels'].shape[0])
            image = batch['observations']['pixels'][random_idx].copy()
            if image.shape[-1] == 6:
                img_0, img_1 = np.split(
                    image, 2, axis=2)
                image = np.concatenate([img_0, img_1], axis=1)
            image_save_dir = os.path.join(os.getcwd(), 'pixels')
            image_save_path = os.path.join(
                image_save_dir, f'observation_{iteration}_batch.png')
            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            io.imsave(image_save_path, image)

        feed_dict = {
            placeholders_flat[key]: batch_flat[key]
            for key in placeholders_flat.keys()
            if key in batch_flat.keys()
        }
        if self._goal_classifier:
            if 'images' in batch.keys():
                images = batch['images']
                goal_sin = batch['observations'][:, -2].reshape((-1, 1))
                goal_cos = batch['observations'][:, -1].reshape((-1, 1))
                goals = np.arctan2(goal_sin, goal_cos)
            else:
                images = batch['observations'][:, :32*32*3].reshape((-1, 32, 32, 3))
            feed_dict[self._placeholders['rewards']] = self._classify_as_goals(images, goals)
        else:
            feed_dict[self._placeholders['rewards']] = batch['rewards']

        if iteration is not None:
            feed_dict[self._placeholders['iteration']] = iteration

        feed_dict[self._placeholders['reward']['running_ext_rew_std']] = self._running_ext_rew_std
        if self._rnd_int_rew_coeff:
            feed_dict[self._placeholders['reward']['running_int_rew_std']] = self._running_int_rew_std

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
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
        # if self._vae:
        #     eval_pixels = feed_dict[self._placeholders['observations']['pixels']]
        #     inputs = (
        #         feed_dict[self._placeholders['actions']],
        #         feed_dict[self._placeholders['observations']['claw_qpos']],
        #         feed_dict[self._placeholders['observations']['goal_index']],
        #         feed_dict[self._placeholders['observations']['last_action']],
        #         eval_pixels,
        #         feed_dict[self._placeholders['observations']['target_xy_position']],
        #         feed_dict[self._placeholders['observations']['target_z_orientation_cos']],
        #         feed_dict[self._placeholders['observations']['target_z_orientation_sin']],
        #     )
        #     test_feed_dict = {
        #         input_ph: input_np
        #         for input_ph, input_np in zip(self._preprocessed_Q_inputs.inputs, inputs)
        #     }
        #     preprocessed_inputs = self._session.run(
        #         self._preprocessed_Q_inputs.output, feed_dict=test_feed_dict)
        #     encoded_pixels = preprocessed_inputs[4]

        #     n_images_to_save = 0
        #     decoded = self._session.run(
        #         self._vae.decode(encoded_pixels, apply_sigmoid=True))

        #     image_save_dir = os.path.join(os.getcwd(), 'vae_reconstructions')
        #     if not os.path.exists(image_save_dir):
        #         os.makedirs(image_save_dir)

        #     if n_images_to_save > 0:
        #         import imageio
        #         sample_idx = np.random.choice(decoded.shape[0], size=n_images_to_save)
        #         concat = np.concatenate([
        #             eval_pixels[sample_idx].astype(np.float32) / 255.,
        #             decoded[sample_idx]], axis=2)

        #         for i in range(n_images_to_save):
        #             image_save_path = os.path.join(
        #                 image_save_dir, f'vae_reconstruction_{iteration}_{i}.png')
        #             imageio.imwrite(image_save_path, concat[i])

        #     from softlearning.models.vae import compute_elbo_loss
        #     # # TODO: Make have this compute the elbo loss, on individual images
        #     loss = self._session.run(compute_elbo_loss(self._vae, eval_pixels))
        #     diagnostics.update({
        #         'vae/elbo_loss': np.mean(loss),
        #     })

            # diagnostics.update({
            #     'vae/elbo_loss_mean': np.mean(loss),
            #     'vae/elbo_loss_max': np.max(loss),
            #     'vae/elbo_loss_min': np.min(loss)
            # })
            # worst_idx = np.argmax(loss)
            # reconstruction_cmp = np.concatenate([
            #     eval_pixels[worst_idx],
            #     skimage.util.img_as_ubyte(decoded[worst_idx])
            # ], axis=1)
            # skimage.io.imsave(
            #     os.path.join(image_save_dir, f'worst_reconstruction_{iteration}.png'),
            #     reconstruction_cmp)

        if 'pixels' in self._policy.preprocessors and \
                self._state_estimator is not None and \
                self._state_estimator.name == 'state_estimator_preprocessor':
            # Train the state estimator on the diagnostic batch
            if self._train_state_estimator_online:
                self._state_estimator.trainable = True
                self._state_estimator.compile(optimizer=self._state_estimator_optimizer, loss='mse')
                train_obs = batch['observations']
                # obs_keys_to_estimate = (
                #     'object_position',
                #     'object_orientation_cos',
                #     'object_orientation_sin',
                # )
                pos = train_obs['object_position'][:, :2]
                pos = normalize(pos, -0.1, 0.1, -1, 1)
                num_samples = pos.shape[0]
                ground_truth_state = np.concatenate([
                    pos,
                    train_obs['object_orientation_cos'][:, 2].reshape((num_samples, 1)),
                    train_obs['object_orientation_sin'][:, 2].reshape((num_samples, 1))
                ], axis=1)

                pixels = train_obs['pixels']
                # ground_truth_state = np.concatenate(
                #     [train_obs[key] for key in obs_keys_to_estimate], axis=1)

                self._state_estimator.fit(
                    x=pixels,
                    y=ground_truth_state,
                    batch_size=32,
                    epochs=self._state_estimator_iters or 1
                )
                self._state_estimator.trainable = False

                self._state_estimator.compile(optimizer=self._state_estimator_optimizer, loss='mse')
                # Copy over weights to the other state estimators
                for Q, Q_target in zip(self._Qs, self._Q_targets):
                    Q.get_layer('state_estimator_preprocessor').set_weights(
                            self._state_estimator.get_weights())
                    Q_target.get_layer('state_estimator_preprocessor').set_weights(
                            self._state_estimator.get_weights())

            # Evaluate on the current diagnostic batch
            state_estimators = [
                self._state_estimator,
                self._Qs[0].get_layer('state_estimator_preprocessor'),
                self._Qs[1].get_layer('state_estimator_preprocessor')
            ]
            state_estimator_descriptors = ('policy', 'Q0', 'Q1')
            # state_estimator = self._state_estimator
            eval_obs = batch['observations']
            # obs_keys_to_estimate = (
            #     'object_position',
            #     'object_orientation_cos',
            #     'object_orientation_sin',
            # )
            pixels = eval_obs['pixels']
            # ground_truth_state = np.concatenate(
            #     [eval_obs[key] for key in obs_keys_to_estimate], axis=1)

            # pos = eval_obs['object_position'][:, :2]

            assert 'object_xy_position' in eval_obs
            pos = eval_obs['object_xy_position']
            from softlearning.models.state_estimation import normalize
            pos = normalize(pos, -0.1, 0.1, -1, 1)
            num_samples = pos.shape[0]
            assert 'object_z_orientation_cos' in eval_obs and 'object_z_orientation_sin' in eval_obs
            labels = np.concatenate([
                pos,
                eval_obs['object_z_orientation_cos'].reshape((num_samples, 1)),
                eval_obs['object_z_orientation_sin'].reshape((num_samples, 1))
                # eval_obs['object_orientation_cos'][:, 2].reshape((num_samples, 1)),
                # eval_obs['object_orientation_sin'][:, 2].reshape((num_samples, 1)),
            ], axis=1)
            preds_per_state_estimator = [
                state_estimator.predict(pixels)
                for state_estimator in state_estimators
            ]

            true_angles = np.arctan2(labels[:, 3], labels[:, 2]) * 180 / np.pi
            for preds, desc in zip(preds_per_state_estimator, state_estimator_descriptors):
                pos_errors_xy = np.abs(labels[:, :2] - preds[:, :2])
                pos_errors = np.linalg.norm(pos_errors_xy, axis=1)
                pos_errors = 15 * pos_errors

                pred_angles = np.arctan2(preds[:, 3], preds[:, 2]) * 180 / np.pi
                angle_errors = angle_distance(true_angles, pred_angles)

                diagnostics.update({
                    f'state_estimator/{desc}/xy_position_error_in_cm': np.mean(
                        pos_errors
                    ),
                    f'state_estimator/{desc}/angle_error_in_degrees_policy': np.mean(
                        angle_errors
                    )
                })

        if self._goal_classifier:
            diagnostics.update({
                'goal_classifier/avg_reward': np.mean(feed_dict[self._rewards_ph])})

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
