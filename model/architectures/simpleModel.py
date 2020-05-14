import tensorflow as tf
import gin


@gin.configurable('simpleModel')
def Architechture(input_size=32, n_filters_in=16, kernel_size=3, neurons=128):

    # f(•)
    inputs = tf.keras.Input((input_size, input_size, 3))

    h_a = tf.keras.layers.Conv2D(filters=n_filters_in * 2 ** 0, kernel_size=kernel_size, strides=(2, 2), padding='same', activation='relu')(inputs)
    h_a = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(h_a)

    h_a = tf.keras.layers.Conv2D(filters=n_filters_in * 2 ** 1, kernel_size=kernel_size, strides=(2, 2), padding='same', activation='relu')(h_a)
    h_a = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(h_a)

    h_a = tf.keras.layers.Conv2D(filters=n_filters_in * 2 ** 2, kernel_size=kernel_size, strides=(2, 2), padding='same',             activation='relu')(h_a)
    h_a = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(h_a)

    h_a = tf.keras.layers.Conv2D(filters=n_filters_in * 2 ** 3, kernel_size=kernel_size, strides=(2, 2), padding='same',             activation='relu')(h_a)
    h_a = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(h_a)





    h_a = tf.keras.layers.GlobalAveragePooling2D()(h_a) #2048
    #h_a = tf.keras.layers.Dense(neurons)(h_a)

    # g(•)
    z_a = tf.keras.layers.Dense(neurons)(h_a)
    z_a = tf.keras.layers.Activation("relu")(z_a)
    z_a = tf.keras.layers.Dense(neurons)(z_a)   #128

    model = tf.keras.Model(inputs=inputs, outputs=[h_a, z_a])

    return model
