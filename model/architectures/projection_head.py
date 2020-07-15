import tensorflow as tf
import gin


@gin.configurable('headModel')
def Architecture_head(mlp_dense1=512, mlp_dense2=128, input_dim=128):

    inputs=tf.keras.Input(input_dim, name="projection_head_input")

    g_a = tf.keras.layers.Dense(mlp_dense1, use_bias=True, name="Head_Dense1")(inputs)
    g_a = tf.keras.layers.BatchNormalization(name="BN_of_Head_Dense1")(g_a)
    g_a = tf.keras.layers.Activation("relu", name="relu_of_Head_Dense1")(g_a)
    g_a = tf.keras.layers.Dense(mlp_dense2, use_bias=False, name="Head_Dense2")(g_a)

    model=tf.keras.Model(inputs=inputs, outputs=g_a)

    return model


# @gin.configurable('headModel')
# class Architecture_head(tf.keras.Model):
#     """General projection head g(•) of SimCLRv1.
#     When using SimCLRv2's additional Dense->BN->Relu it makes more sense to add these layers to encoder.py,
#     as those 3 layers shouldn't be dropped after training, while all layers of this class should be dropped"""
#     def __init__(self, mlp_dense1=128, mlp_dense2=128, input_dim=128):
#         """
#         mlp_dense1: num neurons of first fully-connected-layer
#
#         """
#         super(Architecture_head, self).__init__(name='headModel')
#         self._input_dim = input_dim
#         self._num_neurons_Head_Dense1 = mlp_dense1
#         self._num_neurons_Head_Dense2 = mlp_dense2
#
#         self._input_layer = tf.keras.Input(self._input_dim, name="projection_head_input")
#         self._Head_Dense1 = tf.keras.layers.Dense(self._num_neurons_Head_Dense1, use_bias=True, name="Head_Dense1")
#         self._BN_of_Head_Dense1 = tf.keras.layers.BatchNormalization(name="BN_of_Head_Dense1")
#         self._relu_of_Head_Dense1 = tf.keras.layers.Activation("relu", name="relu_of_Head_Dense1")
#         self._Head_Dense2 = tf.keras.layers.Dense(self._num_neurons_Head_Dense2, use_bias=False, name="Head_Dense2")
#
#     def call(self, inputs):
#
#         inputs=self._input_layer()
#
#         g_a = self._Head_Dense1(inputs)
#         g_a = self._BN_of_Head_Dense1(g_a)
#         g_a = self._relu_of_Head_Dense1(g_a)
#         g_a = self._Head_Dense2(g_a)
#
#         return g_a
#
#     #TypeError: Expected float32 passed to parameter 'y' of op 'Equal', got 'collections' of type 'str' instead.
#     #Error: Expected float32, got 'collections' of type 'str' instead.