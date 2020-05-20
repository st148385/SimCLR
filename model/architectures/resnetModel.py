import tensorflow as tf
import gin


#Gesamtes Model unter Verwendung von resnet-50
@gin.configurable('resnetModel')
def Architechture(input_size=224, n_filters_in=16, kernel_size=3, neurons=128):

    rn50 = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=None)
    rn50.trainable = True

    # Layers
    # f(•)
    inputs = tf.keras.Input((input_size, input_size, 3))
    h_a = rn50(inputs, training=True)
    h_a = tf.keras.layers.GlobalAveragePooling2D()(h_a)
    # g(•)
    z_a = tf.keras.layers.Dense(neurons)(h_a)
    z_a = tf.keras.layers.Activation("relu")(z_a)
    z_a = tf.keras.layers.Dense(neurons)(z_a)

    model=tf.keras.Model(inputs=inputs, outputs=[h_a, z_a])

    return model