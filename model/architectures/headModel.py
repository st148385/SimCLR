import tensorflow as tf
import gin


#Unterer Teil des Models , also projection head g(•) unter Verwendung von resnet-50
@gin.configurable('headModel')
def Architecture_head(input_size=224, n_filters_in=16, kernel_size=3, neurons=128):
    inputs=tf.keras.Input(2048)
    z_a = tf.keras.layers.Dense(neurons)(inputs)
    z_a = tf.keras.layers.Activation("relu")(z_a)
    z_a = tf.keras.layers.Dense(neurons)(z_a)

    model=tf.keras.Model(inputs=inputs, outputs=z_a)

    return model