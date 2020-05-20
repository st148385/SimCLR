import tensorflow as tf
import gin


#Oberer Teil des Models, also encoder f(•), unter Verwendung von resnet-50
@gin.configurable('encoderModel')
def Architecture_encoder(input_size=224, n_filters_in=16, kernel_size=3, neurons=128):
    rn50 = tf.keras.applications.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=None)
    rn50.trainable = True

    # f(•)
    inputs = tf.keras.Input((input_size, input_size, 3))
    h_a = rn50(inputs, training=True)
    h_a = tf.keras.layers.GlobalAveragePooling2D()(h_a)

    model=tf.keras.Model(inputs=inputs, outputs=h_a)

    return model