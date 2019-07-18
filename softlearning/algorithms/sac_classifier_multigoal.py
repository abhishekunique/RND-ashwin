import numpy as np
import tensorflow as tf

from .sac import SAC, td_target
from .vice import VICE
from softlearning.misc.utils import mixup
from softlearning.models.utils import flatten_input_structure

class SACClassifierMultiGoal(SAC):
    def __init__(
            self,
            classifiers,
            goal_example_pools,
            goal_example_validation_pools,
            classifier_lr=1e-4,
            classifier_batch_size=128,
            reward_type='logits',
            n_classifier_train_steps=int(1e4),
            classifier_optim_name='adam',
            mixup_alpha=0.2,
            goal_conditioned=False,
            **kwargs,
    ):
        self._classifiers = classifiers
        self._goal_example_pools = goal_example_pools
        self._goal_example_validation_pools = goal_example_validation_pools

        assert len(classifiers) > 0 and len(classifiers) == len(goal_example_pools) \
                and len(classifiers) == len(goal_example_validation_pools), \
                'Number of goal classifiers needs to match the number of goal pools'

        self._num_goals = len(classifiers)

        self._classifier_lr = classifier_lr
        self._reward_type = reward_type
        self._n_classifier_train_steps = n_classifier_train_steps
        self._classifier_optim_name = classifier_optim_name
        self._classifier_batch_size = classifier_batch_size
        self._mixup_alpha = mixup_alpha
        self._goal_conditioned = goal_conditioned

        super(SACClassifierMultiGoal, self).__init__(**kwargs)

    def _build(self):
        super(SACClassifierMultiGoal, self)._build()
        self._init_classifier_update()

    def _init_placeholders(self):
        super(SACClassifierMultiGoal, self)._init_placeholders()
        self._placeholders['labels'] = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='labels',
        )

    def _get_classifier_training_ops(self):
        if self._classifier_optim_name == 'adam':
            opt_func = tf.train.AdamOptimizer
        elif self._classifier_optim_name == 'sgd':
            opt_func = tf.train.GradientDescentOptimizer
        else:
            raise NotImplementedError

        self._classifier_optimizers = [
            opt_func(
                learning_rate=self._classifier_lr,
                name='classifier_optimizer_' + str(goal)
            )
            for goal in range(self._num_goals)
        ]

        classifier_training_ops = [
            tf.contrib.layers.optimize_loss(
                classifier_loss_t,
                self.global_step,
                learning_rate=self._classifier_lr,
                optimizer=classifier_optimizer,
                variables=classifier.trainable_variables,
                increment_global_step=False,
            )
            for classifier_loss_t, classifier_optimizer, classifier \
                    in zip(self._classifier_losses_t,
                           self._classifier_optimizers,
                           self._classifiers)
       ]

        return classifier_training_ops

    def _init_classifier_update(self):
        classifier_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._classifiers[0].observation_keys
        })

        goal_logits = [classifier(classifier_inputs)
            for classifier in self._classifiers]

        self._classifier_losses_t = [
            tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=self._placeholders['labels']))
            for logits in goal_logits
        ]

        self._classifier_training_ops = self._get_classifier_training_ops()

    def _get_classifier_feed_dicts(self):
        negatives = self.sampler.random_batch(
            self._classifier_batch_size)['observations']

        # Get positives from different goal pools
        rand_positive_indices = [
            np.random.randint(
                goal_examples[next(iter(goal_examples))].shape[0],
                size=self._classifier_batch_size)
            for goal_examples in self._goal_example_pools
        ]
        positives_per_goal = [
            { key: values[rand_positive_ind]
                for key, values in goal_examples.items() }
            for rand_positive_ind, goal_examples \
                in zip(rand_positive_indices, self._goal_example_pools)
        ]

        labels_batch = np.zeros((2 * self._classifier_batch_size, 1))
        labels_batch[self._classifier_batch_size:] = 1.0
        labels_batches = [labels_batch.copy() for _ in range(self._num_goals)]

        observation_batches = [
            {
                key: np.concatenate((negatives[key], positives[key]), axis=0)
                for key in self._classifiers[0].observation_keys
            }
            for positives in positives_per_goal
        ]

        if self._mixup_alpha > 0:
            for goal_index in range(self._num_goals):
                observation_batches[goal_index], labels_batches[goal_index] = mixup(
                    observation_batches[goal_index], labels_batches[goal_index], alpha=self._mixup_alpha)

        feed_dicts = [
            {
                **{
                    self._placeholders['observations'][key]:
                    observations_batch[key]
                    for key in self._classifiers[0].observation_keys
                },
                self._placeholders['labels']: labels_batch
            }
            for observations_batch, labels_batch in zip(observation_batches, labels_batches)
        ]

        return feed_dicts


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

        classifier_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._classifiers[0].observation_keys
        })

        observation_logits_per_classifier = [classifier(classifier_inputs)
            for classifier in self._classifiers]

        goal_index_mask = self._placeholders['observations']['goal_index']
        goal_index_mask = tf.cast(goal_index_mask, dtype=tf.uint8)

        goal_index_masks = [
            tf.cast(goal_index_mask == goal, dtype=tf.bool)
            for goal in range(self._num_goals)
        ]

        # Replace the correct classification logits for the repsective goals
        observation_logits = observation_logits_per_classifier[0]
        for goal in range(1, self._num_goals):
            observation_logits = tf.where(
               goal_index_masks[goal],
               x=observation_logits_per_classifier[goal],
               y=observation_logits
            )

        self._reward_t = observation_logits

        terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

        Q_target = td_target(
            reward=self._reward_scale * self._reward_t,
            discount=self._discount,
            next_value=(1 - terminals) * next_values)

        return Q_target

    def _epoch_after_hook(self, *args, **kwargs):
        if self._epoch == 0:
            for i in range(self._n_classifier_train_steps):
                feed_dicts = self._get_classifier_feed_dicts()
                self._train_classifier_step(feed_dicts)

    def _train_classifier_step(self, feed_dicts):
        losses = []
        for feed_dict, classifier_training_op, classifier_loss_t \
            in zip(feed_dicts, self._classifier_training_ops, self._classifier_losses_t):
            _, loss = self._session.run((
                classifier_training_op, classifier_loss_t
                ), feed_dict)
            losses.append(loss)
        return losses

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(SACClassifierMultiGoal, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        sample_observations = batch['observations']
        goal_indexes = [
            np.random.randint(
                goal_examples[next(iter(goal_examples))].shape[0],
                size=sample_observations[next(iter(sample_observations))].shape[0])
            for goal_examples in self._goal_example_pools
        ]
        goal_observations_per_goal = [
            {
                key: goal_examples[key][goal_index]
                for key in goal_examples.keys()
            }
            for goal_examples, goal_index in zip(self._goal_example_pools, goal_indexes)
        ]
        goal_indexes_validation = [
            np.random.randint(
                goal_examples_validation[next(iter(goal_examples_validation))].shape[0],
                size=sample_observations[next(iter(sample_observations))].shape[0])
            for goal_examples_validation in self._goal_example_validation_pools
        ]
        goal_observations_validation_per_goal = [
            {
                key: goal_examples_validation[key][goal_index]
                for key in goal_examples_validation.keys()
            }
            for goal_examples_validation, goal_index in \
                zip(self._goal_example_validation_pools, goal_indexes_validation)
        ]

        reward_sample, reward_goal, reward_goal_validation, losses = [], [], [], []
        for goal_index in range(self._num_goals):
            goal_observations = goal_observations_per_goal[goal_index]
            goal_observations_validation = goal_observations_validation_per_goal[goal_index]
            reward_sample_goal_observations, classifier_loss = self._session.run(
                (self._reward_t, self._classifier_losses_t[goal_index]),
                feed_dict={
                    **{
                        self._placeholders['observations'][key]: np.concatenate((
                            sample_observations[key],
                            goal_observations[key],
                            goal_observations_validation[key]
                        ), axis=0)
                        for key in self._classifiers[goal_index].observation_keys
                    },
                    self._placeholders['labels']: np.concatenate([
                        np.zeros((sample_observations[next(iter(sample_observations))].shape[0], 1)),
                        np.ones((goal_observations[next(iter(goal_observations))].shape[0], 1)),
                        np.ones((goal_observations_validation[next(iter(goal_observations_validation))].shape[0], 1)),
                    ])
                }
            )
            (reward_sample_observations,
            reward_goal_observations,
            reward_goal_observations_validation) = np.split(
                reward_sample_goal_observations,
                (
                    sample_observations[next(iter(sample_observations))].shape[0],
                    sample_observations[next(iter(sample_observations))].shape[0] \
                        + goal_observations[next(iter(goal_observations))].shape[0]
                ),
                axis=0)
            reward_sample.append(reward_sample_observations)
            reward_goal.append(reward_goal_observations)
            reward_goal_validation.append(reward_goal_observations_validation)
            losses.append(classifier_loss)

        # Add losses/classifier outputs to the dictionary
        diagnostics.update({
            'reward_learning/classifier_loss_' + str(goal): losses[goal]
            for goal in range(self._num_goals)
        })
        diagnostics.update({
            'reward_learning/reward_sample_obs_mean_' + str(goal): reward_sample[goal]
            for goal in range(self._num_goals)
        })
        diagnostics.update({
            'reward_learning/reward_goal_obs_mean_' + str(goal): reward_sample[goal]
            for goal in range(self._num_goals)
        })
        diagnostics.update({
            'reward_learning/reward_goal_obs_validation_mean_' + str(goal): reward_sample[goal]
            for goal in range(self._num_goals)
        })

        return diagnostics

    def _evaluate_rollouts(self, episodes, env):
        """Compute evaluation metrics for the given rollouts."""
        diagnostics = super(SACClassifierMultiGoal, self)._evaluate_rollouts(
            episodes, env)

        learned_reward = self._session.run(
            self._reward_t,
            feed_dict={
                self._placeholders['observations'][name]: np.concatenate([
                    episode['observations'][name]
                    for episode in episodes
                ])
                for name in self._classifiers[0].observation_keys
            })

        diagnostics[f'reward_learning/reward-mean'] = np.mean(learned_reward)
        diagnostics[f'reward_learning/reward-min'] = np.min(learned_reward)
        diagnostics[f'reward_learning/reward-max'] = np.max(learned_reward)
        diagnostics[f'reward_learning/reward-std'] = np.std(learned_reward)

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = super(SACClassifierMultiGoal, self).tf_saveables
        saveables.update({
            '_classifier_optimizer_' + str(goal): self._classifier_optimizers[goal]
            for goal in range(self._num_goals)
        })

        return saveables
