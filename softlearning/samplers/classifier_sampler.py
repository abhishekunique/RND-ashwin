from .simple_sampler import SimpleSampler

class ClassifierSampler(SimpleSampler):
    def __init__(self, algorithm, **kwargs):
        super().__init__(**kwargs)

        self._algorithm = algorithm
        assert hasattr(algorithm, 'get_reward'), (
            'Must implement `get_reward` method to save in replay pool'
        )

    def _process_sample(self,
                        observation,
                        action,
                        reward,
                        terminal,
                        next_observation,
                        info):
        learned_reward = self._algorithm.get_reward(observation)
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [learned_reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation

