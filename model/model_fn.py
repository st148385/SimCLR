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
def gen_encoderModel(Architecture_encoder, input_size, mlp_dense1, mlp_dense2):
    """Model function defining the graph operations.
    :param architecture: architecture module containing Architecture class (tf.keras.Model)
    :param kwargs: additional keywords passed directly to model
    """

    model = Architecture_encoder(input_size, mlp_dense1, mlp_dense2)

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

#resnet50 by calmisential, so just encoder
@gin.configurable()
def gen_resnet50(architecture_custom_resnet_50, input_size, mlp_dense1, mlp_dense2):
    model=architecture_custom_resnet_50()
    model.build( input_shape = (None, input_size, input_size, 3) )

    return model


#Architecture_fullModel
@gin.configurable()
def gen_fullModel(Architecture_fullModel, **kwargs):

    model = Architecture_fullModel(**kwargs)

    return model


#For 'SSL in SimCLR training' idea:
@gin.configurable()
def gen_classifierHead(Architecture_classifierHead, **kwargs):

    model = Architecture_classifierHead(**kwargs)

    return model




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