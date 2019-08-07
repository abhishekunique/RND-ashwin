from .reweighted_replay_pool import ReweightedReplayPool
import numpy as np
from collections import defaultdict


class UniformlyReweightedReplayPool(ReweightedReplayPool):
    def __init__(self,
                 bin_boundaries,
                 bin_obs_keys, # obs keys to bin on
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._bin_boundaries = bin_boundaries
        self._bin_obs_keys = bin_obs_keys
        self._bins = defaultdict(list) # dict: bin_index_tuple->list of sample_inds
        self._reverse_bins = {} # sample_ind -> bin_index_tuple

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
