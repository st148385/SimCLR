import tensorflow as tf
import gin


#Unterer Teil des Models , also projection head g(•) unter Verwendung von resnet-50
@gin.configurable('headModel')
def Architecture_head(input_size=32, mlp_dense1=512, mlp_dense2=128, input_dim=128):
    inputs=tf.keras.Input(input_dim)
    z_a = tf.keras.layers.Dense(mlp_dense1, use_bias=True)(inputs)
    z_a = tf.keras.layers.Activation("relu")(z_a)
    z_a = tf.keras.layers.BatchNormalization()(z_a)
    z_a = tf.keras.layers.Dense(mlp_dense2, use_bias=False)(z_a)

    model=tf.keras.Model(inputs=inputs, outputs=z_a)

    return model

