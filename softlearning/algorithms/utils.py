from copy import deepcopy

from .sac import SAC


def create_SAC_algorithm(variant, *args, sampler, **kwargs):
    policy_params = variant['policy_params']
    algorithm = SAC(
        *args,
        sampler=sampler,
        reparameterize=policy_params['kwargs']['reparameterize'],
        **kwargs)

    return algorithm


ALGORITHM_CLASSES = {
    'SAC': create_SAC_algorithm,
}


def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm
