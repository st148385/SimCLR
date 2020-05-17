import tensorflow as tf
import tensorflow_datasets as tfds

import gin

from model.augmentation_functions import gaussian_filter
from model.augmentation_functions import random_crop_with_resize
from model.augmentation_functions import color_distortion



@gin.configurable                          #Verwendet für darauffolgende Definition die Parameter aus gin.configurable (benutzt wohl config.gin)
def gen_pipeline_train(ds_name='cifar10',#='mnist',
                       tfds_path='~\\tensorflow_datasets',           #/ für Linux, \\ für Windows
                       size_batch=64,
                       b_shuffle=True,
                       size_buffer_cpu=5,
                       shuffle_buffer_size=0,
                       dataset_cache=False,
                       use_random_flip=True,
                       num_parallel_calls=10,
                       x_size=32
                       ):
    # Load and prepare tensorflow dataset
    data, info = tfds.load(name=ds_name,
                           data_dir=tfds_path,
                           split=tfds.Split.TRAIN,
                           shuffle_files=False,
                           with_info=True)

    @tf.function
    def _map_data(*args):           #Wird verwendet von 'dataset = data.map(map_func=_map_data, num_parallel_calls=num_parallel_calls)'
        image = args[0]['image']
        label = args[0]['label']    #Vmtl ist dataset[0]['image'] der Image-Tensor mit shape (224,224,3) und dataset[0][label] z.B. 'dandelion'
        label = tf.one_hot(label, info.features['label'].num_classes)

        # reshape if mnist or fmnist, makes for fun to use mnist at 32x32
        if ds_name in ['mnist', 'fashion_mnist']:
            image = tf.image.resize(image, size=(32, 32))
            x_size=32
        if ds_name in ['tf_flowers']:
            image = tf.image.resize(image, size=(219,219))        #int(219*0.1)=21=odd
            x_size=219                                            # gaussian blur: k_size = int(v1.shape[1] * 0.1)  # kernel size is set to be 10% of the image height/width
        if ds_name in ['cifar10']:
            x_size=32
        # Cast image type and normalize to 0/1
        image = tf.cast(image, tf.float32) / 255.0

        x_i=image
        x_j=image
        return x_i, x_j, label

        #return image, label

    @tf.function
    def _map_augment(*args):        #Wird verwendet von 'dataset = dataset.map(map_func=_map_augment, num_parallel_calls=num_parallel_calls)'
        image = args[0]
        label = args[0]

        x_i, x_j = gaussian_filter(image, image)

        x_i = random_crop_with_resize(x_i, x_size, x_size)
        x_i = color_distortion(x_i)

        x_j = random_crop_with_resize(x_j, x_size, x_size)
        x_j = color_distortion(x_j)


        # Data augmentation takes place here ;) here, one image is processed, not a batch so maybe return patch1, patch2, label
        return x_i, x_j, label

        #return x_j, label

    # Map data
    dataset = data.map(map_func=_map_data, num_parallel_calls=num_parallel_calls)   #Verwende _map_data auf jedes einzelne Element von train_dataset
                                                                                    #num_parallel_calls: num_cpu_threads = 10
    # Cache data
    if dataset_cache:
        dataset = dataset.cache()

    # Shuffle data
    if b_shuffle:
        if shuffle_buffer_size == 0:
            shuffle_buffer_size = info.splits['train'].num_examples
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    # Map Augmentation
    dataset = dataset.map(map_func=_map_augment, num_parallel_calls=num_parallel_calls) #Benutze also aufs ganze dataset _map_augment(dataset[...])

    # Batching
    dataset = dataset.batch(batch_size=size_batch,
                            drop_remainder=True)  # > 1.8.0: use drop_remainder=True
    if size_buffer_cpu > 0:
        dataset = dataset.prefetch(buffer_size=size_buffer_cpu)

    return dataset, info



@gin.configurable
def gen_pipeline_eval(ds_name='cifar10',
                       tfds_path='~/tensorflow_datasets',

                       size_batch=16,
                       b_shuffle=True,
                       size_buffer_cpu=5,
                       shuffle_buffer_size=0,
                       dataset_cache=False,
                       num_parallel_calls=10):

    # Load and prepare tensorflow dataset
    (train_examples, validation_examples), info = tfds.load(
                                                            ds_name,
                                                            data_dir=tfds_path,
                                                            with_info=True,
                                                            as_supervised=True,

                                                            split=['train', 'test']
                                                            )

    num_examples = info.splits['train'].num_examples
    num_classes = info.features['label'].num_classes
    num_validation_examples = info.splits['test'].num_examples

    print("num_examples: ", num_examples, "\nnum_classes: ", num_classes, "\nnum_validation_examples: ", num_validation_examples)

    IMAGE_RES = 224

    # Auf IMAGE_RES formattieren und auf [0,1] normalisieren
    def format_image(image, label):
        image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
        return image, label

    BATCH_SIZE = 32

    train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
    validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

    image_batch, label_batch = next(iter(train_batches.take(1)))    #Damit kann man dann alle Images eines Batches mit den zugehörigen labels plotten
    image_batch = image_batch.numpy()
    label_batch = label_batch.numpy()


    return train_batches, validation_batches


