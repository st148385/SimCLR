import tensorflow as tf
import gin


#Unterer Teil des Models , also projection head g(•) unter Verwendung von resnet-50
@gin.configurable('headModel')
def Architecture_head(mlp_dense1=512, mlp_dense2=128, input_dim=128):

    inputs=tf.keras.Input(input_dim, name="projection_head_input")

    g_a = tf.keras.layers.Dense(mlp_dense1, use_bias=True, name="Head_Dense1")(inputs)
    g_a = tf.keras.layers.Activation("relu", name="relu_of_Head_Dense1")(g_a)
    g_a = tf.keras.layers.BatchNormalization(name="BN_of_Head_Dense1")(g_a)
    g_a = tf.keras.layers.Dense(mlp_dense2, use_bias=False, name="Head_Dense2")(g_a)

    model=tf.keras.Model(inputs=inputs, outputs=g_a)

    return model

