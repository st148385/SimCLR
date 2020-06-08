import tensorflow as tf
import gin


#Unterer Teil des Models , also projection head g(•) unter Verwendung von resnet-50
@gin.configurable('headModel')
def Architecture_head(input_size=32, mlp_dense1=512, mlp_dense2=128):
    inputs=tf.keras.Input(2048)
    z_a = tf.keras.layers.Dense(mlp_dense1)(inputs)
    z_a = tf.keras.layers.Activation("relu")(z_a)
    z_a = tf.keras.layers.Dense(mlp_dense2)(z_a)

    model=tf.keras.Model(inputs=inputs, outputs=z_a)

    return model