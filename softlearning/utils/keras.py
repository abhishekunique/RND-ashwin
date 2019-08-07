import tempfile

import tensorflow as tf


class PicklableKerasModel(object):
    def __getstate__(self):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            tf.keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}

        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()

            loaded_model = tf.keras.models.load_model(
                fd.name, custom_objects={
                    self.__class__.__name__: self.__class__,
                    'PicklableSequential': PicklableSequential,
                    'PicklableModel': PicklableModel})

        self.__dict__.update(loaded_model.__dict__.copy())

    @classmethod
    def from_config(cls, *args, custom_objects=None, **kwargs):
        custom_objects = custom_objects or {}
        custom_objects.update({
            cls.__name__: cls,
            'PicklableSequential': PicklableSequential,
            'PicklableModel': PicklableModel,
            'tf': tf,
        })
        return super(PicklableKerasModel, cls).from_config(
            *args, custom_objects=custom_objects, **kwargs)


class PicklableSequential(PicklableKerasModel, tf.keras.Sequential):
    pass


class PicklableModel(PicklableKerasModel, tf.keras.Model):
    pass
