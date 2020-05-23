import gin
import tensorflow as tf

@gin.configurable()
def gen_model(Architecture, **kwargs):
    """Model function defining the graph operations.
    :param architecture: architecture module containing Architecture class (tf.keras.Model)
    :param kwargs: additional keywords passed directly to model
    """

    model = Architecture(**kwargs)

    return model

#Architecture_encoder
@gin.configurable()
def gen_encoderModel(Architecture_encoder, **kwargs):
    """Model function defining the graph operations.
    :param architecture: architecture module containing Architecture class (tf.keras.Model)
    :param kwargs: additional keywords passed directly to model
    """

    model = Architecture_encoder(**kwargs)

    return model

#Architecture_head
@gin.configurable()
def gen_headModel(Architecture_head, **kwargs):
    """Model function defining the graph operations.
    :param architecture: architecture module containing Architecture class (tf.keras.Model)
    :param kwargs: additional keywords passed directly to model
    """

    model = Architecture_head(**kwargs)

    return model


#Erhält jetzt eine Architecture aus 'architectures', z.B. 'testModel' über architecture.gin: "gen_model.Architecture = @testModel"











#############################################################################################



'''
class SIMCLR_model(tf.keras.Model)
    def __init__(self)
        super(SIMCLR_model, self).__init__()
        self.inputs =     tf.keras.Input(224,224,3)
        self.rn50 =       rn50  #(inputs, training=True)
        self.Pooling =    tf.keras.layers.GlobalAveragePooling2D()
        self.Dense1 =     tf.keras.layers.Dense(512)
        self.Activation = tf.keras.layers.Activation("relu")
        self.Dense2 =     tf.keras.layers.Dense(128)

    def _call_(self, inputs)
        h_a_v1=self.rn50(inputs=inputs, training=True)
        h_a=self.Pooling(h_a_v1)
        
        z=self.Dense1(h_a)
        z=self.Activation(z)
        z=self.Dense2(z)
'''