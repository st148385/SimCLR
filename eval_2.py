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

    model=model_fn.gen_encoderModel()

    restored_checkpoint = train_eval.train_eval(model=model, run_paths=run_paths)


    print(restored_checkpoint)


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

    #TODO: restored_checkpoint (also model) -> KerasLayer
    encoder_h=hub.KerasLayer(restored_checkpoint)

    encoder_h.trainable = False
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
    model = tf.keras.Sequential([
         encoder_h,
         #plausi,
         tf.keras.layers.Dense(10)
         ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # Trainieren unseres eigenen models, über das test-Dataset und validation-Dataset

    EPOCHS = 1
    history = model.fit(train_batches,
                        epochs=EPOCHS,
                        validation_data=validation_batches)

    model.summary()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()






#main()
if __name__ == '__main__':
    device = '0'
    #path_model_id = 'C:\\Users\\Mari\\PycharmProjects\\experiments\\models\\run_2020-05-14T19-00-15\\ckpts\\ckpt-59'  # only to use if starting from existing model
    path_model_id = 'C:\\Users\\Mari\\PycharmProjects\\experiments\\models\\run_2020-05-18T13-13-41'

    # gin config files
    config_names = ['config_eval.gin', 'architecture.gin']

    # start training
    evaluation_train(path_model_id=path_model_id, device=device, config_names=config_names)

