from copy import deepcopy

def get_convnet_preprocessor(name='convnet_preprocessor', **kwargs):
    from softlearning.models.convnet import convnet_model

    preprocessor = convnet_model(name=name, **kwargs)

    return preprocessor

def get_vae_preprocessor(name='vae_preprocessor',
                         encoder_path=None,
                         decoder_path=None,
                         **kwargs):
    from softlearning.models.vae import VAE
    assert encoder_path is not None and decoder_path is not None, (
        "Must specify paths for the encoder/deocder models.")
    vae = VAE(**kwargs)
    vae.encoder.load_weights(encoder_path)
    vae.decoder.load_weights(decoder_path)
    
    preprocessor = vae.get_encoder(trainable=False, name=name)
#     preprocessor.trainable = False
    
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

def get_replication_preprocessor(name='replication_preprocessor', **kwargs):
    from .replication_preprocessor import replication_preprocessor
    preprocessor = replication_preprocessor(name=name, **kwargs)
    return preprocessor

def get_random_nn_preprocessor(name='random_nn_preprocessor', **kwargs):
    from softlearning.models.feedforward import feedforward_model
    preprocessor = feedforward_model(name=name, **kwargs)
    # Don't update weights in this random NN

    import ipdb; ipdb.set_trace()
    preprocessor = tf.stop_gradient(preprocessor)
    return preprocessor

def get_random_matrix_preprocessor(name='random_matrix_preprocessor', **kwargs):
    from .random_matrix_preprocessor import random_matrix_preprocessor
    preprocessor = random_matrix_preprocessor(name=name, **kwargs)
    return preprocessor

def get_feedforward_preprocessor(name='feedforward_preprocessor', **kwargs):
    from softlearning.models.feedforward import feedforward_model

    preprocessor = feedforward_model(name=name, **kwargs)

    return preprocessor

PREPROCESSOR_FUNCTIONS = {
    'ConvnetPreprocessor': get_convnet_preprocessor,
    'FeedforwardPreprocessor': get_feedforward_preprocessor,
    'StateEstimatorPreprocessor': get_state_estimator_preprocessor,
    'VAEPreprocessor': get_vae_preprocessor,
    'ReplicationPreprocessor': get_replication_preprocessor,
    'RandomNNPreprocessor': get_random_nn_preprocessor,
    'RandomMatrixPreprocessor': get_random_matrix_preprocessor,
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
