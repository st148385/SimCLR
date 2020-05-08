import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import gin

tf.print(np.array([[1,5],[2,1]]))

x=51,1
#y=tf.keras.layers.Dense(240)(x)
#tf.print(y)

#print(tfds.list_builders())





#"@tf.function" allows for the intuitive use of both eager execution and AutoGraph, whereby a function can be run using Python syntax initially
#and #then transferred into the equivalent graph code.

#TODO
#1) "Input Pipeline" mit Erhalt von Image1, Image2 (2 versch. Augmentierungen xi, xj von x) in input_fn einbauen.
#2) In training.py meinen Xent Loss von SimCLR einbauen (statt dem Pseudoloss dort).
#3) Was ist yaml? Siehe to_do in utils_misc


#_fkt_name für private
#dataset.map(funktion) führt funktion auf jedes einzelne element von dataset aus