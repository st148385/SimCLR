import os
import logging
import tensorflow as tf
import tensorflow.keras as ks
import gin
import sys
from utils import utils_params


@gin.configurable(blacklist=['model','ds_train', 'ds_train_info', 'run_paths'])
def train(model,
                   ds_train,
                   ds_train_info,
                   run_paths,
                   n_epochs=10,
                   learning_rate=0.001,
                   save_period=1):
    # Generate summary writer
    writer = tf.summary.create_file_writer(os.path.dirname(run_paths['path_logs_train']))
    logging.info(f"Saving log to {os.path.dirname(run_paths['path_logs_train'])}")

    # Define optimizer
    optimizer = ks.optimizers.Adam(learning_rate=learning_rate)

    # Define checkpoints and checkpoint manager
    # manager automatically handles model reloading if directory contains ckpts
    ckpt = tf.train.Checkpoint(net=model,opt=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=run_paths['path_ckpts_train'],
                                              max_to_keep=2, keep_checkpoint_every_n_hours=1)
    ckpt.restore(ckpt_manager.latest_checkpoint)

    if ckpt_manager.latest_checkpoint:
        logging.info(f"Restored from {ckpt_manager.latest_checkpoint}.")
        epoch_start = int(os.path.basename(ckpt_manager.latest_checkpoint).split('-')[1])+1
    else:
        logging.info("Initializing from scratch.")
        epoch_start = 0


    # Define Metrics
    metric_loss_train = ks.metrics.Mean()

    logging.info(f"Training from epoch {epoch_start+1} to {n_epochs}.")
    # use tf variable for epoch passing - so no new trace is triggered
    # if using normal range (instead of tf.range) assign a epoch_tf tensor, otherwise function gets recreated every turn
    epoch_tf = tf.Variable(1, dtype=tf.int32)

    for epoch in range(epoch_start,int(n_epochs)):
        # assign tf variable, graph build doesn't get triggered again
        epoch_tf.assign(epoch)
        logging.info(f"Epoch {epoch + 1}/{n_epochs}: starting training.")

        # Train
        for image1, image2, _ in ds_train:
            # Train on batch
            train_step(model, image1, image2, optimizer, metric_loss_train,epoch_tf)

        # Print summary
        if epoch <=0:
            model.summary()

        # Fetch metrics
        logging.info(f"Epoch {epoch + 1}/{n_epochs}: fetching metrics.")
        loss_train_avg = metric_loss_train.result()

        # Log to tensorboard
        with writer.as_default():
            tf.summary.scalar('loss_train_average', loss_train_avg, step=epoch)

        # Reset metrics after each epoch
        metric_loss_train.reset_states()

        logging.info(f'Epoch {epoch + 1}/{n_epochs}: loss_train_average: {loss_train_avg}')

        # Saving checkpoints
        if epoch % save_period == 0:
            logging.info(f'Saving checkpoint to {run_paths["path_ckpts_train"]}.')
            ckpt_manager.save(checkpoint_number=epoch)
        # write config after everything has been established
        if epoch <= 0:
            gin_string = gin.operative_config_str()
            logging.info(f'Fetched config parameters: {gin_string}.')
            utils_params.save_gin(run_paths['path_gin'], gin_string)

    return 0


@tf.function
def train_step(model, image, optimizer, metric_loss_train,epoch_tf):
    logging.info(f'Trace indicator - train epoch - eager mode: {tf.executing_eagerly()}.')
    with tf.device('/gpu:*'):
        with tf.GradientTape() as tape:
            features, h = model(image,training=True)
            loss = tf.reduce_mean((1.0-h)**2)  # Example loss, TODO
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    # Update metrics
    metric_loss_train.update_state(loss)
    tf.print("Training loss for epoch:", epoch_tf + 1, " and step: ", optimizer.iterations, " - ", loss,
             output_stream=sys.stdout)
    return 0
