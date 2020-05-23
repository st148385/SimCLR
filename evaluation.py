import logging
from model import input_fn, model_fn, train_eval
from utils import utils_params, utils_misc, utils_devices
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub



def evaluation_train(path_model_id = '', device='0', config_names=['config.gin']):

    # generate folder structures
    run_paths = utils_params.gen_run_folder(path_model_id=path_model_id)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # config
    utils_params.inject_gin(config_names, path_model_id=path_model_id)

    # set device params
    utils_devices.set_devices(device)

    # evaluation pipeline:
    train_batches, validation_batches = input_fn.gen_pipeline_eval()

    model = model_fn.gen_encoderModel()

    restored_model = train_eval.load_checkpoint_weights(model=model, run_paths=run_paths)

    # assertion error, falls es nicht geklappt hat:
    #restored_model.assert_consumed()  # lets you know if anything wasn't restored


    print("restored_model:\n", restored_model)   #<tensorflow.python.keras.engine.training.Model object at 0x00000193D7565388>


    encoder_h = hub.KerasLayer(restored_model)


    print("model after model=kerasLayer(model):\n", encoder_h)        #<tensorflow_hub.keras_layer.KerasLayer object at 0x000001106F4D5EC8>




    train_eval.train_evaluation_network_and_plot_result(model_before_dense=encoder_h, train_batches=train_batches, validation_batches=validation_batches)


    ######inside_checkpoint liefert alle trainierten Variablen, WENN path_model_id dem Pfad bis einschließlich ckpt-55 entspricht:
    #
    # inside_checkpoint = tf.train.list_variables( path_model_id )
    # print(inside_checkpoint)   #Gesamter Checkpoint


    # print("Menge der Elemente der Liste: ", len(inside_checkpoint), "und index 742 lautet: ", inside_checkpoint[742], "\n")

    # for k in range (700,761,1):         #742 hat shape (2048)
    #     print(inside_checkpoint[k])


    #encoder_h = tf.train.load_variable(inside_checkpoint[5:13], name='encoder_h')
    #->TypeError: Expected binary or unicode string, got [('net/layer_with_weights-0/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUE', [7, 7, 3, 64]), ('net/layer_with_weights-0/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUE', [7, 7, 3, 64]), ('net/layer_with_weights-0/layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE', [64]), ('net/layer_with_weights-0/layer_with_weights-1/beta/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUE', [64]), ('net/layer_with_weights-0/layer_with_weights-1/beta/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUE', [64]), ('net/layer_with_weights-0/layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE', [64]), ('net/layer_with_weights-0/layer_with_weights-1/gamma/.OPTIMIZER_SLOT/opt/m/.ATTRIBUTES/VARIABLE_VALUE', [64]), ('net/layer_with_weights-0/layer_with_weights-1/gamma/.OPTIMIZER_SLOT/opt/v/.ATTRIBUTES/VARIABLE_VALUE', [64])]

    #for index in range (0,742):
    #    encoder_h = tf.train.load_variable(inside_checkpoint[index], name=encoder_h)

    #assertion error, falls es nicht geklappt hat:
    #feature_extractor.assert_consumed()  # lets you know if anything wasn't restored
    ######

    # TODO
    # Ich erhalte mit inside_checkpoint = tf.train.list_variables(path_model_id) die gespeicherten Variablen als Liste(761 Elemente der Form('net/layer_with_weights-0/layer_with_weights-93/moving_variance/.ATTRIBUTES/VARIABLE_VALUE', [2048]))
    #
    # Dann wollte ich
    # 1) herausfinden, wo der projection head beginnt und somit den projection head wegwerfen:
    # for index in range(0, vor Anfang des projection heads):
    #     encoder_h = tf.funktion(inside_checkpoint[index])
    # 2) und was hierbei "tf.funktion" wäre


    # Verwende jetzt nur h und nicht z
    #h, z = feature_extractor

    # Plausibilitätscheck
    #URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
    #plausi = hub.KerasLayer(URL, input_shape=(224, 224, 3))

    #CHANGE restored_checkpoint (also ein keras model) -> KerasLayer




    '''
    class myModel(tf.keras.Model):
        def __init__(self, top_model):
            super(myModel, self).__init__()
            self.restored_model = top_model
            self.restored_model.trainable=False
            self.dense = tf.keras.layers.Dense(10)
        def call(self, inputs):
            output, _ = self.restored_model(input)
            output = self.dense(output)
            return output

    mymodel = myModel(top_model=restored_checkpoint)
    
    for images, labels in train_batches:
        h_batch, _ = mymodel(images)
        break
    '''







#main()
if __name__ == '__main__':
    device = '0'
    #path_model_id = 'C:\\Users\\Mari\\PycharmProjects\\experiments\\models\\run_2020-05-14T19-00-15\\ckpts\\ckpt-59'  #Für tf.train.list_variables( path_model_id )
    path_model_id = 'C:\\Users\\Mari\\PycharmProjects\\experiments\\models\\run_2020-05-18T15-13-02'

    #Bestes Resultat mit: path_model_id = 'C:\\Users\\Mari\\PycharmProjects\\experiments\\models\\run_2020-05-16T09-23-51'

    # gin config files
    config_names = ['config_eval.gin', 'architecture.gin']

    # start training
    evaluation_train(path_model_id=path_model_id, device=device, config_names=config_names)

