import tensorflow as tf
import gin



@gin.configurable('simpleModel')
def Architechture(input_size=32, n_filters_in=16, kernel_size=3, neurons=128):

    rn50 = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=None,
                                          pooling='avg')
    rn50.training = True

    # Layers
    # f(•)
    inputs = tf.keras.Input((224, 224, 3))
    h_a = rn50(inputs)
    #h_a = tf.keras.layers.GlobalAveragePooling2D()(h_a)
    # g(•)
    z_a = tf.keras.layers.Dense(512)(h_a)
    z_a = tf.keras.layers.Activation("relu")(z_a)
    z_a = tf.keras.layers.Dense(neurons)(z_a)

    model=tf.keras.Model(inputs=inputs, outputs=[h_a, z_a])

    return model