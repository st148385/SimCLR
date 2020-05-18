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
#f und g
@gin.configurable()
def gen_model_gesamt():
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
    z_a = tf.keras.layers.Dense(128)(z_a)

    model=tf.keras.Model(inputs=inputs, outputs=[h_a, z_a])

    return model
#############################################################################################

#Encoder Network f(•)
@gin.configurable()
def gen_encoder_network_f(Architecture = tf.keras.Model):          #**kwargs=inputs
    rn50 = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling='avg')
    rn50.training = True

    #Layers
    inputs=tf.keras.Input((224, 224, 3))
    h_a_resnet = rn50(inputs)
    h_a = tf.keras.layers.GlobalAveragePooling2D()(h_a_resnet)

    model_f = Architecture(inputs, h_a)     #Z.B. model_f = tf.keras.Model(inputs)

    return model_f



#Projection Head g(•)
def gen_projection_head_g(Architecture, **kwargs):

    # TODO || Problem: Wie muss ich model_f aufrufen, um h_a an gen_projection_head_g zu übergeben.
    #Erhalte h_a von model_f
    #h_a = ...
    z_a = tf.keras.layers.Dense(512)(h_a)
    z_a = tf.keras.layers.Activation("relu")(z_a)
    z_a = tf.keras.layers.Dense(128)(z_a)

    model_g = Architecture(**kwargs)

    return model_g


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