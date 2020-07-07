'''
Script to load data from JSON.
'''
import json
import random
import tensorflow as tf

def read_jpg(file, shape, n_classes):

    '''Function to read JPEG Images from GCP GCS bucket.

        Parameters:
            path        - Image files path.
            shape       - Image Shape.
            n_classes   - Number of Classes for OneHot Encoding
        Return Value:
            Processed image and label tensors.
    '''

    image = tf.io.read_file(file[0])
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, [shape, shape], preserve_aspect_ratio=True, method='nearest')
    s = tf.shape(image)
    paddings = [[0, m-s[i]] for (i, m) in enumerate([shape, shape, 3])]
    image = tf.pad(image, paddings, mode='CONSTANT', constant_values=-1)
    image.set_shape((shape, shape, 3))
    out = tf.strings.to_number(file[1])
    out = tf.cast(out, tf.int32)
    ohe = tf.one_hot(out, n_classes)
    return image, ohe


def read_data(json_data, model_shape):

    '''Function to read JSON and return dataset.

        Parameters:
            json_data     - JSON data.
            model_shape   - Image Shape.
        Return Value:
            Processed Tensorflow Dataset.
    '''
    with open(json_data) as f:
        data = json.load(f)
    # data = json.loads(json_data)

    images = [(a['Image URI'], a['Label ID']) for a in data['records']]
    ll = random.sample(images, k=len(images))
    images = ll


    # files = []
    # labels = []
    # for i in  data['records']:
    #     files.append(i['Image URI'])
    #     labels.append(int(i['Label ID'])+1)


    # images = tuple(zip(files, labels))
    # ll = random.sample(images, k=len(images))
    # images = ll
    # files, labels = zip(*images)
    # print(files)
    
    # print(images)

    train_size = int(len(images) * 0.8)
    val_size = len(images) - train_size
    train_data = images[:train_size]
    val_data = images[train_size:]

    classes = set()
    for i in train_data:
        classes.add(i[1])
    print("Classes",classes)

    train_img_ds = tf.data.Dataset.from_tensor_slices((train_data))
    train_img_ds = train_img_ds.map(lambda x: read_jpg(x, model_shape, len(classes)))

    val_img_ds = tf.data.Dataset.from_tensor_slices(val_data)
    val_img_ds = val_img_ds.map(lambda x: read_jpg(x, model_shape, len(classes)))

    return train_img_ds, val_img_ds, train_size, val_size, len(classes)
