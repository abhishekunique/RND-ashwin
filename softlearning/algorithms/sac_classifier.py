import numpy as np
import tensorflow as tf

from .sac import SAC, td_target
from softlearning.misc.utils import mixup
from softlearning.models.utils import flatten_input_structure


class SACClassifier(SAC):
    def __init__(
            self,
            classifier,
            goal_examples,
            goal_examples_validation,
            classifier_lr=1e-4,
            classifier_batch_size=128,
            reward_type = 'logits',
            n_classifier_train_steps=int(1e4),
            classifier_optim_name='adam',
            mixup_alpha=0.2,
            **kwargs,
    ):

        self._classifier = classifier
        self._goal_examples = goal_examples
        self._goal_examples_validation = goal_examples_validation
        self._classifier_lr = classifier_lr
        self._reward_type = reward_type
        self._n_classifier_train_steps = n_classifier_train_steps
        self._classifier_optim_name = classifier_optim_name
        self._classifier_batch_size = classifier_batch_size
        self._mixup_alpha = mixup_alpha
        super(SACClassifier, self).__init__(**kwargs)

    def _build(self):
        super(SACClassifier, self)._build()
        self._init_classifier_update()

    def _init_placeholders(self):
        super(SACClassifier, self)._init_placeholders()
        self._placeholders['labels'] = tf.placeholder(
            tf.float32,
            shape=(None, 2),
            name='labels',
        )

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
            for name in self._classifier.observation_keys
        })
        observation_logits = self._classifier(classifier_inputs)

        if self._reward_type == 'logits':
            self._reward_t = observation_logits
        elif self._reward_type == 'probabilities':
            self._reward_t = tf.nn.sigmoid(observation_logits)
        else:
            raise NotImplementedError(
                f"Unknown reward type: {self._reward_type}")

        terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

        Q_target = td_target(
            reward=self._reward_scale * self._reward_t,
            discount=self._discount,
            next_value=(1 - terminals) * next_values)

        return Q_target

    def _get_classifier_training_op(self):
        if self._classifier_optim_name == 'adam':
            opt_func = tf.train.AdamOptimizer
        elif self._classifier_optim_name == 'sgd':
            opt_func = tf.train.GradientDescentOptimizer
        else:
            raise NotImplementedError

        self._classifier_optimizer = opt_func(
            learning_rate=self._classifier_lr,
            name='classifier_optimizer')

        classifier_training_op = tf.contrib.layers.optimize_loss(
            self._classifier_loss_t,
            self.global_step,
            learning_rate=self._classifier_lr,
            optimizer=self._classifier_optimizer,
            variables=self._classifier.trainable_variables,
            increment_global_step=False,
        )

        return classifier_training_op

    def _init_classifier_update(self):
        logits = self._classifier([self._placeholders['observations']])
        self._classifier_loss_t = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=self._placeholders['labels']))
        self._classifier_training_op = self._get_classifier_training_op()

    def _get_classifier_feed_dict(self):
        negatives = self.sampler.random_batch(
            self._classifier_batch_size)['observations']
        rand_positive_ind = np.random.randint(
            self._goal_examples[next(iter(self._goal_examples))].shape[0],
            size=self._classifier_batch_size)
        positives = {
            key: values[rand_positive_ind]
            for key, values in self._goal_examples.items()
        }

        labels_batch = np.zeros((2*self._classifier_batch_size, 1))
        labels_batch[self._classifier_batch_size:] = 1.0

        observations_batch = {
            key: np.concatenate((negatives[key], positives[key]), axis=0)
            for key in self._classifier.observation_keys
        }

        if self._mixup_alpha > 0:
            observation_batch, labels_batch = mixup(
                observations_batch, labels_batch, alpha=self._mixup_alpha)

        feed_dict = {
            **{
                self._placeholders['observations'][key]:
                observations_batch[key]
                for key in self._classifier.observation_keys()
            },
            self._placeholders['labels']: labels_batch
        }

        return feed_dict

    def _train_classifier_step(self, feed_dict):
        _, loss = self._session.run((
            self._classifier_training_op, self._classifier_loss_t
        ), feed_dict)
        return loss

    def _epoch_after_hook(self, *args, **kwargs):
        if self._epoch == 0:
            for i in range(self._n_classifier_train_steps):
                feed_dict = self._get_classifier_feed_dict()
                self._train_classifier_step(feed_dict)

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        diagnostics = super(SACClassifier, self).get_diagnostics(
            iteration, batch, training_paths, evaluation_paths)

        sample_observations = batch['observations']
        goal_index = np.random.randint(
            self._goal_examples[next(iter(self._goal_examples))].shape[0],
            size=sample_observations[next(iter(sample_observations))].shape[0])
        goal_observations = {
            key: self._goal_examples[key][goal_index]
            for key in self._goal_examples.keys()
        }

        goal_index_validation = np.random.randint(
            self._goal_examples_validation[
                next(iter(self._goal_examples))].shape[0],
            size=sample_observations[next(iter(sample_observations))].shape[0])
        goal_observations_validation = (
            self._goal_examples_validation[goal_index_validation])

        sample_goal_observations = {
            key: np.concatenate((
                sample_observations[key],
                goal_observations[key],
                goal_observations_validation[key]
            ), axis=0)
            for key in sample_observations.keys()
        }

        reward_sample_goal_observations, classifier_loss = self._session.run(
            (self._reward_t, self._classifier_loss_t),
            feed_dict={
                self._placeholders['observations']: sample_goal_observations,
                self._placeholders['labels']: np.concatenate([
                    np.zeros((sample_observations.shape[0], 1)),
                    np.ones((goal_observations.shape[0], 1)),
                    np.ones((goal_observations_validation.shape[0], 1)),
                ])
            }
        )

        # TODO(Avi): Make this clearer. Maybe just make all the vectors
        # the same size and specify number of splits
        (reward_sample_observations,
         reward_goal_observations,
         reward_goal_observations_validation) = np.split(
             reward_sample_goal_observations,
             (
                 sample_observations.shape[0],
                 sample_observations.shape[0] + goal_observations.shape[0]
             ),
             axis=0)

        # TODO(Avi): fix this so that classifier loss is split into train and val
        # currently the classifier loss printed is the mean
        # classifier_loss_train, classifier_loss_validation = np.split(
        #     classifier_loss,
        #     (sample_observations.shape[0]+goal_observations.shape[0],),
        #     axis=0)

        diagnostics.update({
            # 'reward_learning/classifier_loss_train':
            # np.mean(classifier_loss_train),
            # 'reward_learning/classifier_loss_validation':
            # np.mean(classifier_loss_validation),
            'reward_learning/classifier_loss': classifier_loss,
            'reward_learning/reward_sample_obs_mean': np.mean(
                reward_sample_observations),
            'reward_learning/reward_goal_obs_mean': np.mean(
                reward_goal_observations),
            'reward_learning/reward_goal_obs_validation_mean': np.mean(
                reward_goal_observations_validation),
        })

        return diagnostics

    def _evaluate_rollouts(self, episodes, env):
        """Compute evaluation metrics for the given rollouts."""
        diagnostics = super(SACClassifier, self)._evaluate_rollouts(
            episodes, env)

        learned_reward = self._session.run(
            self._reward_t,
            feed_dict={
                self._placeholders['observations'][name]: np.concatenate([
                    episode['observations'][name]
                    for episode in episodes
                ])
                for name in self._classifier.observation_keys
            })

        diagnostics[f'reward_learning/reward-mean'] = np.mean(learned_reward)
        diagnostics[f'reward_learning/reward-min'] = np.min(learned_reward)
        diagnostics[f'reward_learning/reward-max'] = np.max(learned_reward)
        diagnostics[f'reward_learning/reward-std'] = np.std(learned_reward)

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = super(SACClassifier, self).tf_saveables
        saveables.update({
            '_classifier_optimizer': self._classifier_optimizer
        })

        return saveables
