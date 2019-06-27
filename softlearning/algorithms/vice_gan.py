import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

from softlearning.models.utils import flatten_input_structure
from .sac import td_target
from .sac_classifier import SACClassifier


class VICEGAN(SACClassifier):
    """
    A modification on the VICE[1] algorithm which uses a simple discriminator
    (similar to generative adversarial networks).

    References
    ----------
    [1] Variational Inverse Control with Events: A General
    Framework for Data-Driven Reward Definition. Justin Fu, Avi Singh,
    Dibya Ghosh, Larry Yang, Sergey Levine, NIPS 2018.
    """
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
            reward=self._reward_scale * self._reward_t,
            discount=self._discount,
            next_value=(1 - terminals) * next_values)

        return Q_target

    def _epoch_after_hook(self, *args, **kwargs):
        for i in range(self._n_classifier_train_steps):
            feed_dict = self._get_classifier_feed_dict()
            self._train_classifier_step(feed_dict)
