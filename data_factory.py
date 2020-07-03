'''
Script to load data from JSON.
'''
import json
import tensorflow as tf

def read_jpg(path, shape, n_classes):

    '''Function to read JPEG Images from GCP GCS bucket.

        Parameters:
            path        - Image files path.
            shape       - Image Shape.
            n_classes   - Number of Classes for OneHot Encoding
        Return Value:
            Processed image and label tensors.
    '''

    image = tf.io.read_file(path[0])
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, [shape, shape], preserve_aspect_ratio=True, method='nearest')
    s = tf.shape(image)
    paddings = [[0, m-s[i]] for (i, m) in enumerate([shape, shape, 3])]
    image = tf.pad(image, paddings, mode='CONSTANT', constant_values=-1)
    image.set_shape((shape, shape, 3))
    out = tf.strings.to_number(path[1])
    out = tf.cast(out, tf.int32)
    ohe = tf.one_hot(out, n_classes)
    return image, ohe


def read_png(path, shape, n_classes):

    '''Function to read JPEG Images from GCP GCS bucket.

        Parameters:
            path    - Image files path.
            shape   - Image Shape.
        Return Value:
            Processed image tensors.
    '''

    image = tf.io.read_file(path[0])
    image = tf.io.decode_png(image)
    image = tf.image.resize(image, [shape, shape], preserve_aspect_ratio=True, method='nearest')
    s = tf.shape(image)
    paddings = [[0, m-s[i]] for (i, m) in enumerate([shape, shape, 3])]
    image = tf.pad(image, paddings, mode='CONSTANT', constant_values=-1)
    image.set_shape((shape, shape, 3))

    ohe = tf.one_hot(path[1], n_classes)
    return image, ohe


def read_data(json_data, model_shape):

    '''Function to read JSON and return dataset.

        Parameters:
            json_data     - JSON data.
            model_shape   - Image Shape.
        Return Value:
            Processed Tensorflow Dataset.
    '''

    data = json.loads(json_data)

    images = [(a['Image URI'], a['Label ID']) for a in data['records']]

    classes = set()
    for i in images:
        classes.add(i[1])

    img_ds = tf.data.Dataset.from_tensor_slices(images)
    img_ds = img_ds.map(lambda x: read_jpg(x, model_shape, len(classes)))

    return img_ds, len(classes), len(images)
