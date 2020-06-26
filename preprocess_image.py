'''
Script to preprocess PNG images.
'''

import io
import numpy as np
import tensorflow as tf
import PIL
from google.cloud import storage
import tensorflow_io as tfio
from google_cloud_storage import get_blobs

def read_jpg(path, shape):

    '''Function to read JPEG Images from GCP GCS bucket.

        Parameters:
            path    - Image files path.
            shape   - Image Shape.
        Return Value:
            Processed image tensors.
    '''

    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image)
    image.set_shape(shape)
    return image


def read_png(path, shape):

    '''Function to read PNG Images from GCP GCS bucket.

        Parameters:
            path    - Image files path.
            shape   - Image Shape.
        Return Value:
            Processed image tensors.
    '''

    image = tf.io.read_file(path)
    image = tf.io.decode_png(image)
    image.set_shape(shape)
    return image


def reshape_image(image, shape):

    '''Function to reshape images.

        Parameters:
            image    - Image tensor.
        Return Value:
            Reshaped image tensors.
    '''

    image = tf.expand_dims(image, axis=2)
    image.set_shape(shape)
    return image


def get_shape(bucket_name, img_dir):

    '''Function to get shape of image.

        Parameters:
            bucket_name     - The name of the bucket to be instantiated.
            img_dir         - Image directory to be read.
        Return Value:
            Tuple of image shape.
    '''

    image_blobs = sorted(get_blobs(bucket_name, img_dir+"/train/images"))

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blob = bucket.blob(image_blobs[0]).download_as_string()
    byte = io.BytesIO(blob)
    image = PIL.Image.open(byte)

    return np.array(image).shape


def read_images(bucket_name, img_dir, image_extension):

    '''Function to get all images from GCS Bucket directories and read images in tensors

        Parameters:
            bucket_name     - The name of the bucket to be instantiated.
            img_dir         - Image directory to be read.
            image_extensiom - Image File Extension.
        Return Value:
            Tensorflow Dataset of image frame tensors.
    '''

    train_image_blobs = sorted(get_blobs(bucket_name, img_dir+"/train/images"))
    train_image_files = sorted(['gs://'+bucket_name+'/'+i for i in train_image_blobs])

    train_img_ds = tf.data.Dataset.list_files(train_image_files)

    dims = get_shape(bucket_name, img_dir)

    if image_extension in ['.jpg', '.jpeg']:
        train_img_ds = train_img_ds.map(lambda x: read_jpg(x, dims))
    elif image_extension == '.png':
        train_img_ds = train_img_ds.map(lambda x: read_png(x, dims))

    if len(dims) == 2:
        expand_shape = dims+(1,)
        train_img_ds = train_img_ds.map(lambda x: reshape_image(x, expand_shape))

    return train_img_ds
