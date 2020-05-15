import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

import tensorflow_datasets as tfds
import tensorflow_hub as hub

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#Classification Problem erstellen und mein h_a als feature extractor einbauen.
#D.h. ich möchte folgendes model erstellen: model.sequential(feature_extractor_h, dense(10)), mit feature_extractor_h.trainable = False

#Von meinem alten Colab:
##################################################################################################


#Lade das Datenset mit den Bildern herunter:
(train_examples, validation_examples), info = tfds.load(
    'cifar10',
    with_info=True,
    as_supervised=True,
    split=['train', 'test']
    )

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
gehtdas = info.splits['test'].num_examples

print("num_examples: ", num_examples, "\nnum_classes: ", num_classes, "\nnum_validation_examples: ", gehtdas)

#Man bemerkt im folgenden, dass diese Bilder nicht alle die selbe Größe aufweisen:
for i, example_image in enumerate(train_examples.take(3)):
  print("Image {} shape: {}".format(i+1, example_image[0].shape))

IMAGE_RES=224

##################################################################################################


def format_image(image, label):                                     #format_image(image,label) formatiert nun also die Bilder und normalisiert direkt die Pixel-Werte
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 16

train_batches      = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)    #In Python ist "//" Integer-Division. Also teilen und die Kommazahl abschneiden. Z.B. 10//3=3 || 10/3=3.3333
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()



##############################################################################################################
#Teil 3: Transfer Learning (also Verwendung von ImageNet mit unserem Udacity5-Datenset und eben nur 2 möglichen Outputs, bzw. logits: Cat oder Dog)

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
#feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES,3))

#feature_batch = feature_extractor(image_batch)
#print(feature_batch.shape, "(32 is the number of images and 1280 is the number of neurons in the last layer of the ""partial model"" of tensorflow hub)\n\n")
#print("1280 statt 1001 weil ""because we cut off the last layer of MobileNet""")

#Wie im Kurs beschrieben wurde, müssen wir ImageNet "freezen", so dass keine der weights und biases von ImageNet durch unseren Code neu trainiert werden. (Das würde den Sinn von transfer learning kaputt machen)
feature_extractor.trainable = False   #Motto: "We only want to train the layers we add ourselves"

#Definiere nun das tatsächliche model von uns selbst, das eben als eines seiner Layer das heruntergeladene "ImageNet" verwendet.
model = tf.keras.Sequential([
  feature_extractor,            #MobileNet (model das von Profis auf die Datenbank "ImageNet" trainiert wurde)
  tf.keras.layers.Dense(10)               #Unser eigener fully connected layer, durch den es nur noch 2 labels gibt, die wir als Ergebnis erhalten können
])

model.summary()

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

#Trainieren unseres eigenen models, über das test-Dataset und validation-Dataset

EPOCHS = 1
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

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

#Beobachtung: Zu Beginn ist die Genauigkeit des validation sets besser, als die training accuracy. Das ist so, weil MobileNet bereits trainiert war (genauer, siehe Video)