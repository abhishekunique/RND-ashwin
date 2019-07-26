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

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     # Redefine the classifier to force it to be a negative log prob
    #     self._classifier = tf.math.log_sigmoid(self._classifier)

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
        observation_log_p = self._classifier(classifier_inputs)
        curr_policy_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._policy.observation_keys
        })
        curr_actions = self._policy.actions(curr_policy_inputs)
        curr_log_pis = self._policy.log_pis(curr_policy_inputs, curr_actions)

        self._classifier_output_t = observation_log_p
        self._reward_t = observation_log_p - curr_log_pis

        log_pi_log_p_concat = tf.concat([log_pi, log_p], axis=1)
        self._discriminator_output_t = tf.compat.v1.math.softmax(log_pi_log_p_concat)

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
        # DEBUG: Testing with the same negatives pool for each training iteration
        # negatives = type(self._pool.data)(
        #     (key[1], value[:self._classifier_batch_size])
        #     for key, value in self._pool.data.items()
        #     if key[0] == 'observations')
        rand_positive_ind = np.random.randint(
            self._goal_examples[next(iter(self._goal_examples))].shape[0],
            size=self._classifier_batch_size)
        positives = {
            key: values[rand_positive_ind]
            for key, values in self._goal_examples.items()
        }
        labels_batch = np.zeros(
            (2 * self._classifier_batch_size, 2),
            dtype=np.int32)
        labels_batch[:self._classifier_batch_size, 0] = 1
        labels_batch[self._classifier_batch_size:, 1] = 1
        observations_batch = {
            key: np.concatenate((negatives[key], positives[key]), axis=0)
            for key in self._classifier.observation_keys
        }

        if self._mixup_alpha > 0:
            observations_batch, labels_batch = mixup(
                observations_batch, labels_batch, alpha=self._mixup_alpha)

        feed_dict = {
            **{
                self._placeholders['observations'][key]:
                observations_batch[key]
                for key in self._classifier.observation_keys
            },
            self._placeholders['labels']: labels_batch,
        }

        return feed_dict

    def _init_placeholders(self):
        super()._init_placeholders()
        self._placeholders['labels'] = tf.placeholder(
            tf.int32,
            shape=(None, 2),
            name='labels',
        )

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
            tf.compat.v1.losses.softmax_cross_entropy(
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
        goal_index = np.random.randint(
            self._goal_examples[next(iter(self._goal_examples))].shape[0],
            size=sample_observations[next(iter(sample_observations))].shape[0])
        goal_observations = {
            key: values[goal_index] for key, values in self._goal_examples.items()
        }

        goal_index_validation = np.random.randint(
            self._goal_examples_validation[
                next(iter(self._goal_examples_validation))].shape[0],
            size=sample_observations[next(iter(sample_observations))].shape[0])
        goal_observations_validation = {
            key: values[goal_index_validation]
            for key, values in self._goal_examples_validation.items()
        }

        num_sample_observations = sample_observations[
            next(iter(sample_observations))].shape[0]
        sample_labels = np.repeat(((1, 0), ), num_sample_observations, axis=0)

        num_goal_observations = goal_observations[
            next(iter(goal_observations))].shape[0]
        goal_labels = np.repeat(((0, 1), ), num_goal_observations, axis=0)

        num_goal_observations_validation = goal_observations_validation[
            next(iter(goal_observations_validation))].shape[0]
        goal_validation_labels = np.repeat(
            ((0, 1), ), num_goal_observations_validation, axis=0)

        # reward_observations, classifier_outputs, classifier_losses = self._session.run(
        #     (self._reward_t, self._classifier_output_t, self._classifier_loss_t),
        #     feed_dict={
        #         **{
        #             self._placeholders['observations'][key]: values
        #             for key, values in sample_observations.items()
        #         },
        #         self._placeholders['labels']: sample_labels
        #     }
        # )

        reward_negative_observations, classifier_output_negative, discriminator_output_negative, negative_classifier_loss = self._session.run(
            (self._reward_t, self._classifier_output_t, self._discriminator_output_t, self._classifier_loss_t),
            feed_dict={
                **{
                    self._placeholders['observations'][key]: values
                    for key, values in sample_observations.items()
                },
                self._placeholders['labels']: sample_labels
            }
        )

        reward_goal_observations_training, classifier_output_goal_training, discriminator_output_goal_training, goal_classifier_training_loss = self._session.run(
            (self._reward_t, self._classifier_output_t, self._discriminator_output_t, self._classifier_loss_t),
            feed_dict={
                **{
                    self._placeholders['observations'][key]: values
                    for key, values in goal_observations.items()
                },
                self._placeholders['labels']: goal_labels
            }
        )

        reward_goal_observations_validation, classifier_output_goal_validation, discriminator_output_goal_validation, goal_classifier_validation_loss = self._session.run(
            (self._reward_t, self._classifier_output_t, self._discriminator_output_t, self._classifier_loss_t),
            feed_dict={
                **{
                    self._placeholders['observations'][key]: values
                    for key, values in goal_observations_validation.items()
                },
                self._placeholders['labels']: goal_validation_labels
            }
        ) 

        diagnostics.update({
            # classifier loss averaged across the actual training batches
            'reward_learning/classifier_training_loss': np.mean(
                self._training_loss), 
            # classifier loss sampling from the goal image pool
            'reward_learning/classifier_loss_sample_goal_obs_training': np.mean(
                goal_classifier_training_loss), 
            'reward_learning/classifier_loss_sample_goal_obs_validation': np.mean(
                goal_classifier_validation_loss),
            'reward_learning/classifier_loss_sample_negative_obs': np.mean(
                negative_classifier_loss),
            'reward_learning/reward_negative_obs_mean': np.mean(
                reward_negative_observations),
            'reward_learning/reward_goal_obs_training_mean': np.mean(
                reward_goal_observations_training),
            'reward_learning/reward_goal_obs_validation_mean': np.mean(
                reward_goal_observations_validation),
            'reward_learning/classifier_negative_obs_log_p_mean': np.mean(
                classifier_output_negative),
            'reward_learning/classifier_goal_obs_training_log_p_mean': np.mean(
                classifier_output_goal_training),
            'reward_learning/classifier_goal_obs_validation_log_p_mean': np.mean(
                classifier_output_goal_validation),
            'reward_learning/discriminator_output_negative_mean': np.mean(
                discriminator_output_negative),
            'reward_learning/discriminator_output_goal_obs_training_mean': np.mean(
                discriminator_output_goal_training),
            'reward_learning/discriminator_output_goal_obs_validation_mean': np.mean(
                discriminator_output_goal_validation),
   

            # TODO: Figure out why converting to probabilities isn't working
            # 'reward_learning/classifier_negative_obs_prob_mean': np.mean(
            #     tf.nn.sigmoid(reward_negative_observations)),
            # 'reward_learning/classifier_goal_obs_training_prob_mean': np.mean(
            #     tf.nn.sigmoid(reward_goal_observations_training)),
            # 'reward_learning/classifier_goal_obs_validation_prob_mean': np.mean(
            #     tf.nn.sigmoid(reward_goal_observations_validation)),
        })

        return diagnostics
