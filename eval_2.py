import logging
from model import input_fn, model_fn
from model.trainingNEU import train
from utils import utils_params, utils_misc, utils_devices
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def evaluation_train(path_model_id = '', device='0', config_names=['config.gin']):

    # generate folder structures
    run_paths = utils_params.gen_run_folder(path_model_id=path_model_id)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # config
    utils_params.inject_gin(config_names, path_model_id=path_model_id)

    # set device params
    utils_devices.set_devices(device)

    # evalutation pipeline:
    train_batches, validation_batches = input_fn.gen_pipeline_eval()



    # Define model and load its Parameters from a checkpoint
    representations = model_fn.gen_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    ckpt = tf.train.Checkpoint(net=representations, opt=optimizer)
    feature_extractor = ckpt.restore(path_model_id)



    #assertion error, falls es nicht geklappt hat
    #feature_extractor.assert_consumed()  # lets you know if anything wasn't restored

    # Verwende jetzt nur h und nicht z
    h, z = feature_extractor


    #Training NUR mit h, wir werfen z ja weg.
    h.trainable = False

    model = tf.keras.Sequential([h, tf.keras.layers.Dense(10) ])

    model.summary()

    model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    #Trainieren unseres eigenen models, über das test-Dataset und validation-Dataset

    EPOCHS = 1
    history = model.fit(train_batches,
                        epochs=EPOCHS,
                        validation_data=validation_batches)

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
    path_model_id = 'C:\\Users\\Mari\\PycharmProjects\\experiments\\models\\run_2020-05-14T19-00-15\\ckpts\\ckpt-55'  # only to use if starting from existing model
    #path_model_id = 'C:\\Users\\Mari\\PycharmProjects\\experiments\\models\\run_2020-05-14T19-00-15'

    # gin config files
    config_names = ['config.gin', 'architecture.gin']

    # start training
    evaluation_train(path_model_id=path_model_id, device=device, config_names=config_names)

