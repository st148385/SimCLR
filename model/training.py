import os
import logging
import tensorflow as tf
import tensorflow.keras as ks
import gin
import sys
from utils import utils_params
import numpy as np


cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)


def _cosine_simililarity_dim1(x, y):
    v = cosine_sim_1d(x, y)
    return v


def _cosine_simililarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = cosine_sim_2d(tf.expand_dims(x, 1), tf.expand_dims(y, 0))
    return v


def _dot_simililarity_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, C, 1)
    # v shape: (N, 1, 1)
    v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
    return v


def _dot_simililarity_dim2(x, y):
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v

def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)

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
    # if using normal range (instead of tf.range) assign an epoch_tf tensor, otherwise function gets recreated every turn
    epoch_tf = tf.Variable(1, dtype=tf.int32)

    for epoch in range(epoch_start, int(n_epochs)):                 #(start=epoch_start, stop=floor(num_epochs), step=1)
        # assign tf variable, graph build doesn't get triggered again
        epoch_tf.assign(epoch)
        logging.info(f"Epoch {epoch + 1}/{n_epochs}: starting training.")

        # Train
        for image, _ in ds_train:
            # Train on batch
            train_step(model, image, optimizer, metric_loss_train,epoch_tf)

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

# Mask to remove positive examples from the batch of negative samples
negative_mask = get_negative_mask(128)

@tf.function
def train_step(model, image1, image2, optimizer, metric_loss_train,epoch_tf):
    logging.info(f'Trace indicator - train epoch - eager mode: {tf.executing_eagerly()}.')
    with tf.device('/gpu:*'):
        with tf.GradientTape() as tape:
            '''
            batch_size=128
            tau=0.5
            h_i, z_i = model(image1)                    #train_step(model=gen_model_gesamt, image1=image1, iamge2=image2, optimizer=)
            h_j, z_j = model(image2)                    #'gen_model_gesamt' returns 'tf.keras.Model(inputs=inputs, outputs=[h_a, z_a])'

            # normalize projection feature vectors
            z_i = tf.math.l2_normalize(z_i, axis=1)
            z_j = tf.math.l2_normalize(z_j, axis=1)

            # tf.summary.histogram('z_i', z_i, step=optimizer.iterations)
            # tf.summary.histogram('z_j', z_j, step=optimizer.iterations)

            l_pos = _dot_simililarity_dim1(z_i, z_j)
            l_pos = tf.reshape(l_pos, batch_size)
            l_pos = l_pos/tau
            # assert l_pos.shape == (config['batch_size'], 1), "l_pos shape not valid" + str(l_pos.shape)  # [N,1]

            negatives = tf.concat([z_j, z_i], axis=0)

            loss = 0

            for positives in [z_i, z_j]:
                l_neg = _dot_simililarity_dim2(positives, negatives)

                labels = tf.zeros(batch_size, dtype=tf.int32)

                l_neg = tf.boolean_mask(l_neg, negative_mask)
                l_neg = tf.reshape(l_neg, batch_size)
                l_neg = l_neg/tau

                # assert l_neg.shape == (
                #     config['batch_size'], 2 * (config['batch_size'] - 1)), "Shape of negatives not expected." + str(
                #     l_neg.shape)
                logits = tf.concat([l_pos, l_neg], axis=1)  # [N,K+1]
                loss += tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)(y_pred=logits, y_true=labels)

            loss = loss / (2 * batch_size)
            tf.summary.scalar('loss', loss, step=optimizer.iterations)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))










'''
            #features, h = model(image,training=True)
            loss = tf.reduce_mean((1.0-h)**2)  # Example loss, TODO

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))

    # Update metrics
    metric_loss_train.update_state(loss)
    tf.print("Training loss for epoch:", epoch_tf + 1, " and step: ", optimizer.iterations, " - Loss: ", loss,
             output_stream=sys.stdout)
    return 0
