import tensorflow as tf
import gin


#Gesamtes Model unter Verwendung von resnet-50
@gin.configurable('resnetModel')
def Architechture(input_size=32, mlp_dense1=512, mlp_dense2=128):

    rn50 = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(input_size,input_size,3))
    rn50.trainable = True

    # Layers
    # f(•)
    inputs = tf.keras.Input((input_size, input_size, 3))
    h_a = rn50(inputs)
    h_a = tf.keras.layers.GlobalAveragePooling2D()(h_a)
    # g(•)
    z_a = tf.keras.layers.Dense(mlp_dense1)(h_a)
    z_a = tf.keras.layers.Activation("relu")(z_a)
    z_a = tf.keras.layers.Dense(mlp_dense2)(z_a)

    model=tf.keras.Model(inputs=inputs, outputs=[h_a, z_a])

    return model