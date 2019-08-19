from copy import deepcopy


def get_convnet_preprocessor(name='convnet_preprocessor', **kwargs):
    from softlearning.models.convnet import convnet_model

    preprocessor = convnet_model(name=name, **kwargs)

    return preprocessor

def get_state_estimator_preprocessor(
        name='state_estimator_preprocessor',
        state_estimator_path='/home/justinvyu/dev/softlearning-vice/softlearning/models/state_estimator_model_random_data_50_epochs.h5',
        **kwargs
    ):
    from softlearning.models.state_estimation import state_estimator_model
    preprocessor = state_estimator_model(**kwargs)

    print('Loading model weights...')
    preprocessor.load_weights(state_estimator_path)

    # Set all params to not-trainable
    preprocessor.trainable = False
    preprocessor.compile(optimizer='adam', loss='mean_squared_error')

    preprocessor.summary()
    return preprocessor


def get_feedforward_preprocessor(name='feedforward_preprocessor', **kwargs):
    from softlearning.models.feedforward import feedforward_model

    preprocessor = feedforward_model(name=name, **kwargs)

    return preprocessor


PREPROCESSOR_FUNCTIONS = {
    'ConvnetPreprocessor': get_convnet_preprocessor,
    'FeedforwardPreprocessor': get_feedforward_preprocessor,
    'StateEstimatorPreprocessor': get_state_estimator_preprocessor,

    None: lambda *args, **kwargs: None
}


def get_preprocessor_from_params(env, preprocessor_params, *args, **kwargs):
    if preprocessor_params is None:
        return None

    preprocessor_type = preprocessor_params.get('type', None)
    preprocessor_kwargs = deepcopy(preprocessor_params.get('kwargs', {}))

    if preprocessor_type is None:
        return None

    preprocessor = PREPROCESSOR_FUNCTIONS[
        preprocessor_type](
            *args,
            **preprocessor_kwargs,
            **kwargs)

    return preprocessor


def get_preprocessor_from_variant(variant, env, *args, **kwargs):
    preprocessor_params = variant['preprocessor_params']
    return get_preprocessor_from_params(
        env, preprocessor_params, *args, **kwargs)
