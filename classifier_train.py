"""
Script to train the model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
import gc
from collections import namedtuple
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from absl import app
from absl import flags
from absl import logging
from models import InceptionV3Model
import data_factory
import google_cloud_storage
warnings.filterwarnings("ignore")


def train_model(params):
    '''Function to train the model and upload the model on GCP GCS Bukcet.

        Parameters:
            params      - The parameters required to fetch the data and train the model
        Return Value:
            None
    '''

    print(params)
    data, num_classes, data_size = data_factory.read_data(params.data_dir, 224)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    t_ds = data.repeat()
    t_ds = t_ds.batch(1)
    t_ds = t_ds.prefetch(buffer_size=AUTOTUNE)

    model_architect = InceptionV3Model((224, 224, 3), num_classes)
    model = model_architect.get_model()

    e_s = EarlyStopping(monitor='categorical_accuracy', mode='max', verbose=1, patience=5)
    m_c = ModelCheckpoint('best_model.h5', monitor='categorical_accuracy',
                          mode='max', save_best_only=True, verbose=1)

    class MyCustomCallback(tf.keras.callbacks.Callback):
        '''Class to return custom callback while training model.

            Parameters:
                tf.keras.callbacks.Callback     - Abstract base class used to build new callbacks.
        '''

        def on_epoch_end(self, epoch, logs=None):
            '''Function to collect garbage after each epoch.

                Parameters:
                    epoch   - Training epoch number
                    logs    - Log verbose to be monitored
            '''
            gc.collect()

    train_size = int(data_size * 0.8)
    val_size = data_size - train_size

    val_steps = int(val_size/(train_size/params.steps_per_epoch))

    train_data = t_ds.take(train_size)
    val_data = t_ds.skip(train_size)

    model.fit(train_data, validation_data=val_data, epochs=params.epochs,
              steps_per_epoch=params.steps_per_epoch,
              validation_steps=val_steps, verbose=1,
              callbacks=[MyCustomCallback(), e_s, m_c])

    bucket_name = params.model_dir.split('/')[2]

    dir_struct = ''
    for i in params.model_dir.split('/')[3:]:
        dir_struct += '/'+i

    google_cloud_storage.upload_blob(
        bucket_name, 'best_model.h5', dir_struct+'/best_model.h5')


def _get_params_from_flags(flags_obj):
    '''Function to get parameters dictionary from flags

        Parameters:
            flags_obj   - The parameters passed during calling train script.
        Return Value:
            None
    '''

    flags_overrides = {
        'model_dir': flags_obj.model_dir,
        'data_dir': flags_obj.data_dir,
        'run_eagerly': flags_obj.run_eagerly,
        'resize': flags_obj.resize,
        'multi_worker': flags_obj.multi_worker,
        'epochs': flags_obj.epochs,
        'steps_per_epoch': flags_obj.steps_per_epoch
    }
    params = namedtuple('Struct', flags_overrides.keys()
                        )(*flags_overrides.values())
    return params


def define_classifier_flags():
    '''Function defines common flags for image classification

        Parameters:
            None
        Return Value:
            None
    '''

    flags.DEFINE_string(
        'data_dir',
        default=None,
        help='The location of the input data.')
    flags.DEFINE_string(
        'model_dir',
        default=None,
        help='The location to save the model')
    flags.DEFINE_bool(
        'run_eagerly',
        default=False,
        help='Use eager execution and disable autograph for debugging.')
    flags.DEFINE_bool(
        'resize',
        default=True,
        help='Resize image to the size of the model without disturbing the aspect ratio')
    flags.DEFINE_bool(
        'multi_worker',
        default=False,
        help='Enable multi-worker strategy')
    flags.DEFINE_integer(
        'epochs',
        default=100,
        help='The number of epochs for which the model needs to be trained')
    flags.DEFINE_integer(
        'steps_per_epoch',
        default=50,
        help='The number of batches per epoch for which the model needs to be trained')


def run(flags_obj):
    '''Function to run the train job.

        Parameters:
            flags_obj   - The parameters passed during calling train script.
        Return Value:
            None
    '''

    params = _get_params_from_flags(flags_obj)
    train_model(params)


def main(_):
    '''Main Function.

        Parameters:
            flags_obj   - The parameters passed during calling train script.
        Return Value:
            None
    '''

    run(flags.FLAGS)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_classifier_flags()
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('model_dir')

    app.run(main)
