"Script to load models for Multi-Label Image Classification"

from __future__ import absolute_import, division, print_function

from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from absl import logging


class InceptionV3Model:
    '''Class to create InceptionV3 Model object.

            Parameters:
                input_shape     - Shape of the Input Layer.
                output_shape	- Number of classes to be given as output.
    '''

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def get_model(self):
        '''Function to load InceptionV3 Model.

            Parameters:
                input_shape     - Shape of the Input Layer.
                output_shape	- Number of classes to be given as output.
            Return Value:
                Tensorflow Keras InceptionV3 Model.
        '''

        inception = InceptionV3(
            weights='imagenet', include_top=False, input_shape=self.input_shape)
        for layer in inception.layers:
            layer.trainable = False
        x_layer = inception.output
        x_layer = Flatten()(x_layer)
        x_layer = Dense(1024, activation='relu',
                        kernel_regularizer=regularizers.l2(0.005))(x_layer)
        x_layer = Dropout(0.2)(x_layer)

        if self.output_shape == 2:
            predictions = Dense(self.output_shape,
                                activation='sigmoid')(x_layer)
        else:
            predictions = Dense(self.output_shape,
                                activation='softmax')(x_layer)

        model = Model(inputs=inception.input, outputs=predictions)
        model.compile(optimizer=SGD(lr=0.001, momentum=0.9),
                      loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        logging.info(model.summary())

        return model