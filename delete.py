import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import gin


######
import os

path = r"C:\\Users\Mari\PycharmProjects\experiments\models"
os.chdir(path)
files = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)

oldest = files[0]
newest = files[-1]

print("Oldest:", oldest)
print("Newest:", newest)
print("All by modified oldest to newest:", files)
######


tf.print(np.array([[1,5],[2,1]]))

x=51,1
#y=tf.keras.layers.Dense(240)(x)
#tf.print(y)

#print(tfds.list_builders())

c = tf.constant([1.0, 2.0])
c=tf.math.l2_normalize(c)       #Vektor c / sqrt(1**2 + 2**2) = [[1,2]]/sqrt(5) = [[1/sqrt(5),2/sqrt(5)]] = [[0.4472,0.8944]]
print(c,"\n")

d = tf.constant( [ [[1.0, 2.0],[5,-1]], [[1,2],[5,-1]] ] )
print( tf.math.l2_normalize(d, axis=1) )



'''
J1 = tf.ones( (128,1) )
J2 = tf.ones( (128,1) )

print(tf.concat( (J1,J2), axis=0 ))
print(tf.concat( (J1,J2), axis=1 ))
'''
'''
#AssertionError:
meineliste = ['element']    #meineliste: ['element']

assert len(meineliste) >= 1
meineliste.pop()    #meineliste: []

assert len(meineliste) >= 1
#-> AssertionError, weil len(meineliste) < 1, wodurch [len(meineliste) >= 1]=False
'''

#"@tf.function" allows for the intuitive use of both eager execution and AutoGraph, whereby a function can be run using Python syntax initially
#and #then transferred into the equivalent graph code.

#
#3) Was ist yaml? Siehe to_do in utils_misc
'''
#dict = {'note mathe':1,'note fach':2}

#print(dict['note mathe'])
'''

#_fkt_name für private
#dataset.map(funktion) führt funktion auf jedes einzelne element von dataset aus
'''
import tensorflow as tf
import datetime
import os
import yaml
import time
import shutil

print(tf.__version__)
from models.cnn_small import SmallCNN
from models.resnet_simclr import ResNetSimCLR
from utils.losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2
from utils.helpers import get_negative_mask, gaussian_filter
from augmentation.transforms import read_images, distort_simclr, read_record, distort_with_rand_aug

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

config = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
input_shape = eval(config['input_shape'])

train_dataset = tf.data.TFRecordDataset('./data/tfrecords/train.tfrecords')
train_dataset = train_dataset.map(lambda x: read_record(x, input_shape),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(distort_simclr, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(gaussian_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.repeat(config['epochs'])
train_dataset = train_dataset.shuffle(4096)
train_dataset = train_dataset.batch(config['batch_size'], drop_remainder=True)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(3e-4)

# model = SmallCNN(out_dim=config['out_dim'])
model = ResNetSimCLR(input_shape=input_shape, out_dim=config['out_dim'])

# Mask to remove positive examples from the batch of negative samples
negative_mask = get_negative_mask(config['batch_size'])

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join('logs', current_time, 'train')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


@tf.function
def train_step(xis, xjs):
    with tf.GradientTape() as tape:
        ris, zis = model(xis)
        rjs, zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        # tf.summary.histogram('zis', zis, step=optimizer.iterations)
        # tf.summary.histogram('zjs', zjs, step=optimizer.iterations)

        l_pos = sim_func_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (config['batch_size'], 1))
        l_pos /= config['temperature']
        # assert l_pos.shape == (config['batch_size'], 1), "l_pos shape not valid" + str(l_pos.shape)  # [N,1]

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(config['batch_size'], dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (config['batch_size'], -1))
            l_neg /= config['temperature']

            # assert l_neg.shape == (
            #     config['batch_size'], 2 * (config['batch_size'] - 1)), "Shape of negatives not expected." + str(
            #     l_neg.shape)
            logits = tf.concat([l_pos, l_neg], axis=1)  # [N,K+1]
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * config['batch_size'])
        tf.summary.scalar('loss', loss, step=optimizer.iterations)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


with train_summary_writer.as_default():
    for xis, xjs in train_dataset:
        # print(tf.reduce_min(xis), tf.reduce_max(xjs))
        # fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=False)
        # axs[0, 0].imshow(xis[0])
        # axs[0, 1].imshow(xis[1])
        # axs[1, 0].imshow(xis[2])
        # axs[1, 1].imshow(xis[3])
        # plt.show()
        # start = time.time()
        train_step(xis, xjs)
        # end = time.time()
        # print("Total time per batch:", end - start)

model_checkpoints_folder = os.path.join(train_log_dir, 'checkpoints')
if not os.path.exists(model_checkpoints_folder):
    os.makedirs(model_checkpoints_folder)
    shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

model.save_weights(os.path.join(model_checkpoints_folder, 'model.h5'))
'''