import os
import logging
import tensorflow as tf
import tensorflow.keras as ks
import gin
import sys
from utils import utils_params


@gin.configurable #(whitelist=[eval_epochs])
def train_evaluation_network_and_plot_result(model_before_dense, train_batches, validation_batches, eval_epochs=2,
                                             dataset_num_classes=10, plot_folder='E:\\Mari\\Texte\\DL\\SimCLR_ckpts\\Plots\\plotname'):
    #Freeze pre-trained encoder h(•)
    model_before_dense.trainable = False

    #Build eval_model
    model = tf.keras.Sequential([
         model_before_dense,
         tf.keras.layers.Dense(dataset_num_classes)
         ])

    #Set opt, loss, metrics
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    #Training
    history = model.fit(train_batches,
                        epochs=eval_epochs,
                        validation_data=validation_batches)

    model.summary()

    #Plot result
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(eval_epochs)

    import matplotlib
    matplotlib.use('TkAgg')  # qt may not work on server
    from matplotlib import pyplot as plt

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
    plt.savefig(plot_folder)
    plt.show()




#@gin.configurable(blacklist=['model', 'run_paths'])
def load_checkpoint_weights(model,
                            run_paths,
                            learning_rate=0.001):
    # Generate summary writer
    writer = tf.summary.create_file_writer(os.path.dirname(run_paths['path_logs_train']))
    logging.info(f"Saving log to {os.path.dirname(run_paths['path_logs_train'])}")

    # Define optimizer
    optimizer = ks.optimizers.Adam(learning_rate=learning_rate)

    # Define checkpoints and checkpoint manager
    # manager automatically handles model reloading if directory contains ckpts
    ckpt = tf.train.Checkpoint(net=model,opt=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=run_paths['path_ckpts_train'], #C:\Users\Mari\PycharmProjects\experiments\models\run_2020-05-14T19-00-15['path_ckpts_train']
                                              max_to_keep=2, keep_checkpoint_every_n_hours=1)
    ckpt.restore(ckpt_manager.latest_checkpoint)  #Nimmt sich wohl [model_checkpoint_path: "ckpt-55"] aus dem Ordner


    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    return model

