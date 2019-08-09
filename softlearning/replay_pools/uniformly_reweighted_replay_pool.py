from .reweighted_replay_pool import ReweightedReplayPool
import numpy as np
from collections import defaultdict


class UniformlyReweightedReplayPool(ReweightedReplayPool):
    def __init__(self,
                 bin_boundaries,
                 bin_obs_keys, # obs keys to bin on
                 inverse_proportion_exploration_bonus_scaling=0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._bin_boundaries = bin_boundaries
        self._bin_obs_keys = bin_obs_keys
        self._bins = defaultdict(list) # dict: bin_index_tuple->list of sample_inds
        self._reverse_bins = {} # sample_ind -> bin_index_tuple
        self._inverse_proportion_exploration_bonus_scaling = inverse_proportion_exploration_bonus_scaling

    def _set_sample_weights(self, batch, batch_indices):
        observation = batch['observations']
        bin_inds = []
        for i, batch_ind in enumerate(batch_indices):
            bin_ind = []
            bin_dim = 0
            for key in self._bin_obs_keys:
                for j in range(observation[key].shape[1]):
                    bin_ind.append(
                        np.digitize(
                            observation[key][i, j],
                            self._bin_boundaries[bin_dim]))
                    bin_dim += 1
            bin_ind = tuple(bin_ind)
            bin_inds.append(bin_ind)

            self._bins[bin_ind].append(batch_ind)
            if self._size == self._max_size: # samples are starting to get deleted
                old_bin_ind = self._reverse_bins[batch_ind]
                self._bins[old_bin_ind].remove(batch_ind)
            self._reverse_bins[batch_ind] = bin_ind
        bin_inds = set(bin_inds)

        delta_norm_constant = 0
        for bin_ind in bin_inds:
            sample_inds = self._bins[bin_ind]
            delta_norm_constant -= np.sum(self._unnormalized_weights[sample_inds])
            self._unnormalized_weights[sample_inds] = 1 / len(sample_inds)
            delta_norm_constant += np.sum(self._unnormalized_weights[sample_inds])
        self._normalization_constant += delta_norm_constant

    def random_batch(self, batch_size, field_name_filter=None, **kwargs):
        """ Modify random training batch by adding an exploration bonus to the rewards. """
        random_indices = self.random_indices(batch_size)
        batch = self.batch_by_indices(
            random_indices, field_name_filter=field_name_filter, **kwargs)
        if self._inverse_proportion_exploration_bonus_scaling:
            ## Add bonus to rewards
            inverse_proportions = np.array([self._normalization_constant/len(self._bins[self._reverse_bins[i]]) for i in random_indices])
            inverse_proportions = inverse_proportions.reshape(-1, 1)
            batch['rewards'] += inverse_proportions * self._inverse_proportion_exploration_bonus_scaling
        return batch
