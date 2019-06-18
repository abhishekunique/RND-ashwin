import numpy as np
import tensorflow as tf

from .sac import td_target
from .sac_classifier import SACClassifier
from softlearning.misc.utils import mixup
from softlearning.models.utils import flatten_input_structure


class VICE(SACClassifier):
    """Varitational Inverse Control with Events (VICE)

    References
    ----------
    [1] Variational Inverse Control with Events: A General
    Framework for Data-Driven Reward Definition. Justin Fu, Avi Singh,
    Dibya Ghosh, Larry Yang, Sergey Levine, NIPS 2018.
    """

    # TODO Avi This class has  a lot of code repeated from SACClassifier due
    # to labels having different dimensions in the two classes, but this can
    # likely be fixed
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
        self._reward_t = observation_logits

        terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

        Q_target = td_target(
            reward=self._reward_scale * observation_logits,
            discount=self._discount,
            next_value=(1 - terminals) * next_values)

        return Q_target

    def _get_classifier_feed_dict(self):
        negatives = self.sampler.random_batch(
            self._classifier_batch_size
        )['observations']
        rand_positive_ind = np.random.randint(
            self._goal_examples.shape[0], size=self._classifier_batch_size)
        positives = self._goal_examples[rand_positive_ind]

        labels_batch = np.zeros(
            (2 * self._classifier_batch_size, 2),
            dtype=np.int32)
        labels_batch[:self._classifier_batch_size, 0] = 1
        labels_batch[self._classifier_batch_size:, 1] = 1
        observation_batch = np.concatenate([negatives, positives], axis=0)

        if self._mixup_alpha > 0:
            observation_batch, labels_batch = mixup(
                observation_batch, labels_batch, alpha=self._mixup_alpha)

        feed_dict = {
            self._placeholders['observations']: observation_batch,
            self._placeholders['labels']: labels_batch
        }

        return feed_dict

    def _init_classifier_update(self):
        classifier_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._classifier.observation_keys
        })
        log_p = self._classifier(classifier_inputs)
        policy_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._policy.observation_keys
        })
        sampled_actions = self._policy.actions(policy_inputs)
        log_pi = self._policy.log_pis(policy_inputs, sampled_actions)
        log_pi_log_p_concat = tf.concat([log_pi, log_p], axis=1)

        self._classifier_loss_t = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(
                self._placeholders['labels'],
                log_pi_log_p_concat,
            )
        )
        self._classifier_training_op = self._get_classifier_training_op()

    def _epoch_after_hook(self, *args, **kwargs):
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
        goal_index = np.random.randint(self._goal_examples.shape[0],
                                       size=sample_observations.shape[0])
        goal_observations = self._goal_examples[goal_index]

        goal_index_validation = np.random.randint(
            self._goal_examples_validation.shape[0],
            size=sample_observations.shape[0])
        goal_observations_validation = \
            self._goal_examples_validation[goal_index_validation]

        sample_goal_observations = np.concatenate(
            (sample_observations, goal_observations, goal_observations_validation),
            axis=0)

        label_1 = np.zeros((sample_observations.shape[0],2), dtype=np.int32)
        label_2 = np.zeros((goal_observations.shape[0],2))
        label_3 = np.zeros((goal_observations_validation.shape[0],2))

        label_1[:, 0] = 1
        label_2[:, 1] = 1
        label_3[:, 1] = 1

        reward_sample_goal_observations, classifier_loss = self._session.run(
            [self._reward_t, self._classifier_loss_t],
            feed_dict={
                self._placeholders['observations']: sample_goal_observations,
                self._placeholders['labels']: np.concatenate((
                    label_1, label_2, label_3))
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
                 sample_observations.shape[0]+goal_observations.shape[0]
             ),
             axis=0)

        # TODO(Avi): fix this so that classifier loss is split into train and val
        # currently the classifier loss printed is the mean
        # classifier_loss_train, classifier_loss_validation = np.split(
        #     classifier_loss,
        #     (sample_observations.shape[0]+goal_observations.shape[0],),
        #     axis=0)

        diagnostics.update({
            # 'reward_learning/classifier_loss_train': np.mean(classifier_loss_train),
            # 'reward_learning/classifier_loss_validation': np.mean(classifier_loss_validation),
            'reward_learning/classifier_loss': classifier_loss,
            'reward_learning/reward_sample_obs_mean': np.mean(
                reward_sample_observations),
            'reward_learning/reward_goal_obs_mean': np.mean(
                reward_goal_observations),
            'reward_learning/reward_goal_obs_validation_mean': np.mean(
                reward_goal_observations_validation),
        })

        return diagnostics
