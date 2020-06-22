import datetime
import os
import logging
import tensorflow as tf
import tensorflow.keras as ks
import gin
import sys
from utils import utils_params
import matplotlib.pyplot as plt


@gin.configurable
def custom_train_evaluation_network(simclr_encoder_h, train_batches, validation_batches,
                                             n_epochs=2, dataset_num_classes=10,
                                             plot_folder='E:\\Mari\\Texte\\DL\\SimCLR_ckpts\\Plots\\plotname', run_paths='~/experiments'):

    # Tensorboard
    train_log_dir = os.path.dirname(run_paths['path_logs_train']) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/train'
    val_log_dir = os.path.dirname(run_paths['path_logs_train']) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/validation'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    logging.info(f"Saving log to {os.path.dirname(run_paths['path_logs_train'])}")  # <path_model_id>\\logs\\run.log

    # Define additional models
    denseModel = tf.keras.Sequential( tf.keras.layers.Dense(dataset_num_classes) )

    # Define loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Define optimizer
    optimizer = ks.optimizers.Adam(learning_rate=0.001)

    # Define Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    epoch_tf = tf.Variable(1, dtype=tf.int32)

    # Freeze pre-trained encoder h(•)
    simclr_encoder_h.trainable = False

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            a = simclr_encoder_h(images, training=True)
            b = denseModel(a)

            loss = loss_object(labels, b)
        gradients = tape.gradient(loss, [simclr_encoder_h.trainable_variables, denseModel.trainable_variables])

        optimizer.apply_gradients(zip(gradients[0], simclr_encoder_h.trainable_variables))
        optimizer.apply_gradients(zip(gradients[1], denseModel.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, b)

    @tf.function
    def validation_step(images, labels):
        a = simclr_encoder_h(images, training=False)
        b = denseModel(a)

        val_loss_inStep = loss_object(labels, b)

        val_loss(val_loss_inStep)
        val_acc(labels, b)


    for epoch in range(0,int(n_epochs)):
        # assign tf variable, graph build doesn't get triggered again
        epoch_tf.assign(epoch)
        logging.info(f"Epoch {epoch + 1}/{n_epochs}: starting training.")

        # Reset Metrics at the start of every epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_acc.reset_states()

        # Train
        for image, label in train_batches:
            train_step(image, label)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        # Validation
        for image, label in validation_batches:
            validation_step(image, label)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_acc.result(), step=epoch)

        # Print summary
        if epoch <=0:
                print("Summary des Encoders f(•):")
                simclr_encoder_h.summary()
                print(f"Summary des Dense({dataset_num_classes}):")
                denseModel.summary()


        # Fetch metrics
        logging.info(f"Epoch {epoch + 1}/{n_epochs}: fetching metrics.")

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Validation_Loss: {}, Validation_Accuracy: {}'
        logging.info(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              val_loss.result(),
                              val_acc.result() * 100))


        # write config after everything has been established
        if epoch <= 0:
            gin_string = gin.operative_config_str()
            logging.info(f'Fetched config parameters: {gin_string}.')
            utils_params.save_gin(run_paths['path_gin'], gin_string)    # <path_model_id>\\config_operative.gin

            # Check trainable (also visible in model summary)
            print("simclr_encoder_h was trainable =", simclr_encoder_h.trainable, "as seen in summary above")

            # Best val_acc with corresponding epoch
#            max_val_acc = max(val_acc)
#            corresponding_index = 1 + val_acc.index(max_val_acc)

#            print(
#                f"Best validation accuracy in {n_epochs} evaluation epochs was {max_val_acc} after epoch {corresponding_index}/{eval_epochs}")

    return 0
























@gin.configurable #(whitelist=[eval_epochs])
def train_evaluation_network_and_plot_result(simclr_encoder_h, train_batches, validation_batches,
                                             eval_epochs=2, dataset_num_classes=10,
                                             plot_folder='E:\\Mari\\Texte\\DL\\SimCLR_ckpts\\Plots\\plotname', run_paths='~/experiments'):



    #Freeze pre-trained encoder h(•)
    simclr_encoder_h.trainable = False

    #Build eval_model
    model = tf.keras.Sequential([
         simclr_encoder_h,     #model_before_dense is encoderModel with or w/o loaded ckpts, depending on path_model_id = '' or path_model_id = '~/path'
         tf.keras.layers.Dense(dataset_num_classes)
         ])

    #Set opt, loss, metrics
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    #Tensorboard
    log_dir = os.path.dirname(run_paths['path_logs_train']) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #Training
    history = model.fit(train_batches,
                        epochs=eval_epochs,
                        validation_data=validation_batches,
                        callbacks=[tensorboard_callback])

    model.summary()

    #Plot result
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(eval_epochs)

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

    #Check trainable (also visible in model summary)
    print("simclr_encoder_h was trainable =", simclr_encoder_h.trainable, "as seen in summary above")

    #Best val_acc with corresponding epoch
    max_val_acc = max(val_acc)
    corresponding_index = 1 + val_acc.index(max_val_acc)

    print(f"Best validation accuracy in {eval_epochs} evaluation epochs was {max_val_acc} after epoch {corresponding_index}/{eval_epochs}")

    #Get config
    gin_string = gin.operative_config_str()
    logging.info(f'Fetched config parameters: {gin_string}.')
    utils_params.save_gin(run_paths['path_gin_eval'], gin_string)  # <path_model_id>\\config_operative.gin

#@gin.configurable(blacklist=['model', 'run_paths'])
def load_checkpoint_weights(model,
                            run_paths,
                            learning_rate=0.001):

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

