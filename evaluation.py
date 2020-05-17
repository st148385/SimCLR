import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

import tensorflow_datasets as tfds
import tensorflow_hub as hub

import logging

from model import model_fn
from model.train_eval import train_eval
from utils import utils_params

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#Classification Problem erstellen und mein h_a als feature extractor einbauen.
#D.h. ich möchte folgendes model erstellen: model.sequential(feature_extractor_h, dense(10)), mit feature_extractor_h.trainable = False

#Von meinem alten Colab:
##################################################################################################

#VGL cats_vs_dogs und cifar10!
(train_examples, validation_examples), info = tfds.load(
    'cifar10',
    with_info=True,
    as_supervised=True,
    split=['train', 'test'])       #cifar10 besitzt 'test' mit 10k examples und 'train' mit 50k examples
    #split=['train[:80%]', 'train[80%:]'])

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
num_validation_examples = info.splits['test'].num_examples

print("num_examples: ", num_examples, "\nnum_classes: ", num_classes, "\nnum_validation_examples: ", num_validation_examples)

for i, example_image in enumerate(train_examples.take(3)):
  print("Image {} shape: {}".format(i+1, example_image[0].shape))

# #cats vs dog hat kein 'test'!
# (train_examples, validation_examples), info = tfds.load(
#     'cats_vs_dogs',
#     with_info=True,
#     as_supervised=True,
#     split=['train[:80%]', 'train[80%:]'])       #cats vs dogs hat nur train
#
#
# num_examples = info.splits['train'].num_examples
# num_classes = info.features['label'].num_classes
# #gehtdas = info.splits['test'].num_examples
#
# print("num_examples (val und test): ", num_examples, "\nnum_classes: ", num_classes)
#
# #Man bemerkt im folgenden, dass diese Bilder nicht alle die selbe Größe aufweisen:
# for i, example_image in enumerate(train_examples.take(3)):
#   print("Image {} shape: {}".format(i+1, example_image[0].shape))


IMAGE_RES=224

##################################################################################################



#Formatieren für cats vs dogs o.ä. und normalisieren
def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

train_batches      = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)    #In Python ist "//" Integer-Division. Also teilen und die Kommazahl abschneiden. Z.B. 10//3=3 || 10/3=3.3333
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)




#Plausibilitätscheck mit mobilenet_v2
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES,3))

#####

path_model_id = 'C:\\Users\\Mari\\PycharmProjects\\experiments\\models\\run_2020-05-14T19-00-15'
run_paths = utils_params.gen_run_folder(path_model_id=path_model_id)

model=model_fn.gen_model()
restored_checkpoint = train_eval(model=model, run_paths=run_paths)
restored_checkpoint.trainable=False
h=hub.KerasLayer(restored_checkpoint)
#####



#freeze
feature_extractor.trainable = False

#model
model = tf.keras.Sequential([
  #feature_extractor,
  h,
  tf.keras.layers.Dense(10)
])

model.summary()

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


#Trainieren unseres eigenen models, über das test-Dataset und validation-Dataset

EPOCHS = 6
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)




#Plot Ergebnis
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

