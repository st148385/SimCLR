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
    # x shape: (N, 1, C)            (N=batch_size)
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
    # if using normal range (instead of tf.range) assign a epoch_tf tensor, otherwise function gets recreated every turn
    epoch_tf = tf.Variable(1, dtype=tf.int32)

    for epoch in range(epoch_start,int(n_epochs)):
        # assign tf variable, graph build doesn't get triggered again
        epoch_tf.assign(epoch)
        logging.info(f"Epoch {epoch + 1}/{n_epochs}: starting training.")

        # Train
        for image, image2, _ in ds_train:
            # Train on batch
            train_step(model, image, image2, optimizer, metric_loss_train,epoch_tf, batch_size=128, tau=0.2)

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

#@gin.configurable

#@tf.function
def train_step(model, image, image2, optimizer, metric_loss_train, epoch_tf, batch_size, tau):
    logging.info(f'Trace indicator - train epoch - eager mode: {tf.executing_eagerly()}.')

    # Mask to remove positive examples from the batch of negative samples



    with tf.device('/gpu:*'):
        with tf.GradientTape() as tape:

            h_i, z_i = model(image)  # train_step(model=gen_model_gesamt, image1=image1, image2=image2, optimizer=)
            h_j, z_j = model(image2)  # 'gen_model_gesamt' returns 'tf.keras.Model(inputs=inputs, outputs=[h_a, z_a])'


            #print("h_i:\n",h_i.shape)       #(128,128)
            #print("z_i:\n",z_i.shape)       #(128,128)

            #tf.print("tf.print -> h_i:\n", h_i.shape)
            #tf.print("tf.print -> z_i:\n", z_i.shape)

            ###Shapes: z_i=(128,128), h_i=(128,2048)

            # normalize projection feature vectors
            z_i = tf.math.l2_normalize(z_i, axis=1)         #Vektor z = z/||z||
            z_j = tf.math.l2_normalize(z_j, axis=1)

            #print("z_i:\n", z_i.shape)      #(128,128)
            ###Shape: z_i=(128,128)

            # tf.summary.histogram('z_i', z_i, step=optimizer.iterations)
            # tf.summary.histogram('z_j', z_j, step=optimizer.iterations)

            l_pos = _dot_simililarity_dim1(z_i, z_j)    #l_pos = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2)), d.h. shape(z_i) wird (•,1,•) und shape(z_j) wird (•,•,1)

            #print("l_pos:\n", l_pos.shape)  #(128,1,1)

            l_pos = tf.reshape(l_pos, (batch_size, 1) ) #l_pos erhält shape=(128,1) -> column vector

            #print("l_pos:\n", l_pos.shape)  #(128,1)

            l_pos = l_pos / tau

            #print("l_pos:\n", l_pos.shape)  #(128,1)

            assert l_pos.shape == (batch_size, 1), "l_pos shape ist falsch!" + str(l_pos.shape)  # [N,1]

            #print("z_i:\n", z_i.shape)  #(128,128)
            #print("z_j:\n", z_j.shape)  #(128,128)

            negatives = tf.concat([z_j, z_i], axis=0)   #concat: Wenn z_j shape (a,b) und z_i shape (a,b) haben, hat negatives shape (a+a,b)
                                                        #Mit axis=1 hätte im selben Beispiel negatives die shape (a,b+b)

            #print("negatives:\n", negatives.shape)  #(256,128)

            loss = 0

            for positives in [z_i, z_j]:
                #print("negatives:\n", negatives.shape) #256,128
                #print("positives:\n", positives.shape) #128,128        #positives ist 1 Mal z_i, und dann 1 Mal z_j   (z_i und z_j entsprechen halt dem ganzen Batch)
                l_neg = _dot_simililarity_dim2(positives, negatives)    #l_neg = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)

                #print("l_neg:\n", l_neg.shape)         #128,256

                #VORSICHT: vor der tf.boolean_mask hat l_neg 128*256=32768 Elemente.
                #Danach ist aber die komplette Diagonale von get_negative_mask(batch_size) 'False'. Dadurch werden von l_neg durch tf.boolean_mask so viele
                #Elemente entfernt, wie es eben Elemente in der Diagonalen gibt.
                # (Alle Stellen mit 'False' werden aus l_neg gelöscht, sodass sie nicht berechnet werden müssen)

                labels = tf.zeros(batch_size, dtype=tf.int32)

                l_neg = tf.boolean_mask( l_neg,  get_negative_mask(batch_size) )        #negative_mask = get_negative_mask(batch_size) # alle elemente der diagnole vervwerfen

                #print("l_neg:\n", l_neg.shape)          #(32512,)       #127*256=32512, bzw. 128*254=32512

                l_neg = tf.reshape(l_neg, (batch_size, -1) )

                #Error: reshape (32512,) -> (128,1). Stattdessen jetzt (32512,) -> (128,32512/128)=(128,254)

                #print("l_neg:\n", l_neg.shape)     #(128,254)


                l_neg = l_neg / tau

                # assert l_neg.shape == (
                #     config['batch_size'], 2 * (config['batch_size'] - 1)), "Shape of negatives not expected." + str(
                #     l_neg.shape)

                #print("l_pos:\n", l_pos.shape)         #(128,1)        (2 Images der 2N=2Batch_Size Images bilden ein positive pair)
                #print("l_neg:\n", l_neg.shape)         #(128,254)      (Die verbleibenden 2N-2=254 Images liefern die negatives)

                logits = tf.concat([l_pos, l_neg], axis=1)  # [N,K+1]   #"logits": "This Tensor is the quantity that is being mapped to probabilities by the Softmax"

                #print("logits:\n", logits)     #(128,255)
                ###Shape: logits=(128,2)

                loss += tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)(y_pred=logits, y_true=labels)

                #print("loss:\n", loss.shape)   #()

                ###Shape: loss=()   (loss soll ein Skalar sein und ist hier auch ein Skalar)

            loss = loss / (2 * batch_size)
            tf.summary.scalar('loss', loss, step=optimizer.iterations)
        print(loss)
        #print(model.trainable_variables.shape)         #AttributeError: 'list' object has no attribute 'shape'

        gradients = tape.gradient(loss, model.trainable_variables)          #error 08.05. || 18:12 Uhr  TODO
        # ValueError: Cannot reshape a tensor with 128 elements to shape [32512] (32512 elements) for 'Reshape_16' (op:
        # 'Reshape') with input shapes: [128], [1] and with input tensors computed as partial shapes: input[1] = [32512].


        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Update metrics
    metric_loss_train.update_state(loss)
    tf.print("Training loss for epoch:", epoch_tf + 1, " and step: ", optimizer.iterations, " - ", loss,
             output_stream=sys.stdout)
    return 0
