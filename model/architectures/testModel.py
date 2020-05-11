import tensorflow as tf
import tensorflow.keras as ks
import gin
import numpy as np


@gin.configurable('testModel')
def Architechture(input_size=224):
    rn50 = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=None,
                                          pooling='avg')
    rn50.training = True

    # Layers
    # f(•)
    inputs = tf.keras.Input((input_size, input_size, 3))
    h_a = rn50(inputs)
    #h_a = tf.keras.layers.GlobalAveragePooling2D()(h_a)
    # g(•)
    z_a = tf.keras.layers.Dense(512)(h_a)
    z_a = tf.keras.layers.Activation("relu")(z_a)
    z_a = tf.keras.layers.Dense(128)(z_a)

    model=tf.keras.Model(inputs=inputs, outputs=[h_a, z_a])

    return model
