from .flexible_replay_pool import FlexibleReplayPool
from .simple_replay_pool import normalize_observation_fields

import numpy as np
from gym.spaces import Dict

def random_int_with_variable_range(mins, maxs):
    result = np.floor(np.random.uniform(mins, maxs)).astype(int)
    return result

class RelabelReplayPool(FlexibleReplayPool):
    def __init__(self, observation_space, action_space, *args,
        relabel_probability=0.8, relabel_reward=1.0, **kwargs):

        self._observation_space = observation_space
        self._action_space = action_space
        self._relabel_probability = relabel_probability
        self._relabel_reward = relabel_reward

        observation_fields = normalize_observation_fields(observation_space)
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have
        # to worry about termination conditions.
        observation_fields.update({
            'next_' + key: value
            for key, value in observation_fields.items()
        })

        fields = {
            **observation_fields,
            **{
                'actions': {
                    'shape': self._action_space.shape,
                    'dtype': 'float32'
                },
                'rewards': {
                    'shape': (1, ),
                    'dtype': 'float32'
                },
                # self.terminals[i] = a terminal was received at time i
                'terminals': {
                    'shape': (1, ),
                    'dtype': 'bool'
                },
                # self.terminals[i] = a terminal was received at time i
                'timesteps_left_in_episode': {
                    'shape': (1, ),
                    'dtype': 'int32'
                },
            }
        }

        super(RelabelReplayPool, self).__init__(
            *args, fields_attrs=fields, **kwargs)

    def add_samples(self, samples):
        if isinstance(self._observation_space, Dict):
            raise NotImplementedError

        field_names = list(samples.keys())
        num_samples = samples[field_names[0]].shape[0]

        index = np.arange(
            self._pointer, self._pointer + num_samples) % self._max_size

        for field_name in self.field_names:
            if field_name == 'timesteps_left_in_episode':
                action_values = samples.get('actions', default_value)
                episode_length = action_values.shape[0]
                timesteps_left_in_episode = np.arange(
                    episode_length, 0, -1, dtype=np.int32)
                self.fields[field_name][index] = np.expand_dims(
                    timesteps_left_in_episode, axis=1)
            else:
                default_value = (
                    self.fields_attrs[field_name].get('default_value', 0.0))
                values = samples.get(field_name, default_value)
                assert values.shape[0] == num_samples
                self.fields[field_name][index] = values

        self._advance(num_samples)
    

    def batch_by_indices(self,
                         indices,
                         field_name_filter=None,
                         observation_keys=None):
        if isinstance(self._observation_space, Dict):
            raise NotImplementedError

        batch_size = indices.shape[0]

        field_names = self.field_names
        if field_name_filter is not None:
            field_names = self.filter_fields(
                field_names, field_name_filter)

        initial_batch = {
            field_name: self.fields[field_name][indices]
            for field_name in field_names
        }

        assert self.fields['observations'].shape[1] % 2 == 0

        # import IPython; IPython.embed()
        future_indices_max = np.copy(initial_batch['timesteps_left_in_episode'])
        future_indices_max = np.squeeze(future_indices_max, axis=1)
        future_indices_max += indices
        future_sample_inds = random_int_with_variable_range(
            indices, future_indices_max)

        future_sample_inds = future_sample_inds % self._size

        obs_dim = int(self.fields['observations'].shape[1]/2)
        future_obs = self.fields['observations'][future_sample_inds][:, :obs_dim]

        relabeled_batch = {
            'observations': np.copy(initial_batch['observations']),
            'next_observations': np.copy(initial_batch['next_observations']),
            'rewards': np.copy(initial_batch['rewards'])
        }

        #assumes the reward to be 1/0
        relabeled_batch['observations'][:, obs_dim:] = future_obs
        relabeled_batch['next_observations'][:, obs_dim:] = future_obs
        relabeled_batch['rewards'][:, obs_dim:] = self._relabel_reward

        resample_index = (
            np.random.rand(batch_size) < self._relabel_probability)
        where_resampled = np.where(resample_index)

        for key in ['observations', 'next_observations', 'rewards']:
            initial_batch[key][where_resampled] = relabeled_batch[key][where_resampled]

        return initial_batch

    def terminate_episode(self):
        pass