import numpy as np

from .multi_sac_classifier import MultiSACClassifier


class MultiVICEGAN(MultiSACClassifier):
    def _epoch_after_hook(self, *args, **kwargs):
        losses_per_classifier = [[] for _ in range(self._num_goals)]
        for i in range(self._n_classifier_train_steps):
            classifier_index = i % self._num_goals
            feed_dict = self._get_classifier_feed_dict(classifier_index)
            losses_per_classifier[classifier_index].append(
                self._train_classifier_step(classifier_index, feed_dict))
        self._training_losses_per_classifier = [
            np.concatenate(loss, axis=-1) for loss in losses_per_classifier]
