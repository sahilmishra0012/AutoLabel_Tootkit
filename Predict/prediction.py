'''
Script for Predictions
'''
import os
import tensorflow as tf
import numpy as np
from google.cloud import storage

global model


def read_jpg(file, shape):

    '''Function to read JPEG Images from GCP GCS bucket.

        Parameters:
            path        - Image files path.
            shape       - Image Shape.
        Return Value:
            Processed image and label tensors.
    '''

    image = tf.io.read_file(file)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, [shape, shape], preserve_aspect_ratio=True, method='nearest')
    s = tf.shape(image)
    paddings = [[0, m-s[i]] for (i, m) in enumerate([shape, shape, 3])]
    image = tf.pad(image, paddings, mode='CONSTANT', constant_values=-1)
    image.set_shape((shape, shape, 3))
    return image


def load_preprocess_data(files):

    '''Function to read image paths and load images.

        Parameters:
            files        - List of image files paths for predictions.
        Return Value:
            List of image arrays.
    '''

    images = []
    for i in files:
        images.append(read_jpg(i, 224))
    return images


def download_and_load_model(model_path):
    
    '''Function to read model path, download and load_model.

        Parameters:
            model_path        - Model path of latest model.
        Return Value:
            Model Loading Status.
    '''
    
    global model
    blobs_list = []
    splits = model_path.split('/')
    bucket_name = splits[2]
    dir_name = ""
    for i in splits[3:]:
        dir_name = dir_name+"/"+i
    print(dir_name[1:])
    if not os.path.exists(dir_name[1:]):
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=dir_name[1:])
        for blob in blobs:
            print(blob.name)
            blobs_list.append(blob.name)
            filename = blob.name
            if '/' in filename:
                create_dir = filename.rsplit('/', 1)[0]
                os.makedirs(create_dir, mode=0o777, exist_ok=True)
                blob.download_to_filename(filename)  # Download
        print("Model Downloaded")
    else:
        print("Model Already Downloaded")
        try:
            model = tf.keras.models.load_model(dir_name[1:])
            return "Model Loaded"
        except:
            return "Model Loading Failed"


def predict_on_data(data):

    '''Function to make predictions.

        Parameters:
            data        - List of image arrays for predictions.
        Return Value:
            List of predictions.
    '''

    global model
    prediction_list = []
    for i in data:
        prediction_list.append(model.predict(np.expand_dims(i, axis=0)))
    json_data = {'probabilities': str(prediction_list)}
    return json_data
