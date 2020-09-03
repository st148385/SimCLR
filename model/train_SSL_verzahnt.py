import os
import logging
import tensorflow as tf
import tensorflow.keras as ks
import gin
import sys
from utils import utils_params
import numpy as np
import math
from math import pi as pi
cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)


def _cosine_simililarity_dim1(x, y):    #unused
    v = cosine_sim_1d(x, y)
    return v


def _cosine_simililarity_dim2(x, y):    #unused
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


class lr_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Takes:\n
    lr_max: Global maximum of learning-rate function\n
    overallSteps: overall number of training-steps, i.e. steps per epoch * epochs\n
    warmupDuration: Duration of warmup-phase - compared to full duration - as percentages in decimal-format,
    i.e. warmupDuration=0.05 results in 5% linear warmup and 95% cosine decay. (Default is 0.1 meaning 10%)\n

    Returns:
    learning-rate function
    """
    def __init__(self, lr_max, overallSteps, warmupDuration=0.1):
        super(lr_schedule, self).__init__()

        self.justCosineDecay = False

        self.lr_max = lr_max
        self.lr_max = tf.cast(self.lr_max, tf.float32)

        self.overallSteps = overallSteps

        if warmupDuration==0:
            self.justCosineDecay=True
            print(f"Just using cosine decay over all {self.overallSteps} steps")
        else:
            self.warmupPercent = warmupDuration
            self.N = (4 / self.warmupPercent) - 4
            self.warmup_steps = math.ceil(self.overallSteps * self.warmupPercent)
            print(f"Using linear warmup for the first {self.warmup_steps} steps, then cosine decay till the end")


    def __call__(self, step):

        if self.justCosineDecay==True:
            return abs(self.lr_max * tf.math.cos( (0.5*pi*step) / (self.overallSteps) ))

        else:
            cos_decay = (tf.math.cos((2 * pi * (step - self.warmup_steps)) / (self.N * self.warmup_steps)))

            lin_warmup = step * (self.warmup_steps ** -1)

            return abs( (self.lr_max) * tf.math.minimum(cos_decay, lin_warmup) )


def supervised_nt_xent_loss(z, y, temperature=0.5, base_temperature=0.07):
    '''
    Taken from github: https://github.com/wangz10/contrastive_loss/blob/master/losses.py

    Supervised normalized temperature-scaled cross entropy loss.
    A variant of Multi-class N-pair Loss from (Sohn 2016)
    Later used in SimCLR (Chen et al. 2020, Khosla et al. 2020).
    Implementation modified from:
        - https://github.com/google-research/simclr/blob/master/objective.py
        - https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
    '''
    batch_size = tf.shape(z)[0]
    contrast_count = 1
    anchor_count = contrast_count
    y = tf.expand_dims(y, -1)

    # mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
    #     has the same class as sample i. Can be asymmetric.
    mask = tf.cast(tf.equal(y, tf.transpose(y)), tf.float32)
    anchor_dot_contrast = tf.divide(
        tf.matmul(z, tf.transpose(z)),
        temperature
    )
    # # for numerical stability
    logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
    logits = anchor_dot_contrast - logits_max
    # # tile mask
    logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - \
        tf.math.log((1e-6) + tf.reduce_sum(exp_logits, axis=1, keepdims=True))     #Needs addition of (1e-6), otherwise
                                      # neither log_prob, nor mask_sum will contain NaNs, BUT log_prob * mask_sum will!

    # compute mean of log-likelihood over positive
    # this may introduce NaNs due to zero division,
    # when a class only has one example in the batch
    mask_sum = tf.reduce_sum(mask, axis=1)

    ### Check why loss is nan
    # if tf.reduce_any(tf.math.is_nan(mask)) == True:
    #     raise ValueError("mask contains a NaN")
    # if tf.reduce_any(tf.math.is_nan(mask_sum)) == True:
    #     raise ValueError("mask_sum contains a NaN")
    # if tf.reduce_any(tf.math.is_nan(log_prob)) == True:
    #     raise ValueError("log_prob contains a NaN")
    # if tf.reduce_any(tf.math.is_nan(mask * log_prob)) == True:
    #     raise ValueError("mask * log_prob contains a NaN")
    # if tf.reduce_any(tf.math.is_nan(tf.reduce_sum(mask * log_prob, axis=1))) == True:
    #     raise ValueError("tf.reduce_sum(mask * log_prob, axis=1) contains a NaN")
    # if tf.reduce_any(tf.math.is_nan(tf.reduce_sum(mask * log_prob, axis=1)[mask_sum > 0])) == True:
    #     raise ValueError("tf.reduce_sum(mask * log_prob, axis=1)[mask_sum > 0] contains a NaN")
    ###

    mean_log_prob_pos = tf.reduce_sum(
        mask * log_prob, axis=1)[mask_sum > 0] / (mask_sum[mask_sum > 0])

    # loss
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    # loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))
    loss = tf.reduce_mean(loss)

    # if tf.math.is_nan(loss) == True:
    #     raise ValueError("loss is NaN, even though neither of 'mask', 'log_prob', 'mask_sum' or "
    #                      "'mask * log_prob, axis=1)[mask_sum > 0]' contains NaNs, so 'mask_sum[mask_sum > 0]' must "
    #                      "contain a zero, which makes no sense.")

    return loss


@gin.configurable(blacklist=['model','model_head','model_gesamt','ds_train', 'ds_train_info', 'run_paths'])     #Eine Variable in der blacklist kann über die config.gin KEINEN Wert erhalten.
def train(model, model_head, model_classifierHead, another_model_head, model_gesamt,
          ds_train,
          ds_train_info,
          train_batches,
          test_batches,
          run_paths,
          n_epochs=10,
          learning_rate_noScheduling=0.001,
          lr_max_ifScheduling=0.001,
          save_period=1,
          size_batch=128,
          tau=0.5,
          use_2optimizers=False,
          use_split_model=True,
          use_learning_rate_scheduling=True,
          warmupDuration=0.1,
          SSLeveryNepochs=10
          ):
    """Executes the Training Loop of SimCLR. So it trains the simclr's encoder h(•) and projection head g(•).
    Also tries to load ckpts of started trainings from run_paths and saves trained checkpoints to run_paths.

    Pass both parts of split model and the overall model (all 3 models), so parameter "use_split_model" can be used.
    """





    #Use model_gesamt as model, if use_split_model==False
    if use_split_model == False:
        model = model_gesamt
        raise ValueError("'use_split_model = False' doesnt make sense in semi-supervised simclr training.")

    # Generate summary writer
    writer = tf.summary.create_file_writer(os.path.dirname(run_paths['path_logs_train']))
    logging.info(f"Saving log to {os.path.dirname(run_paths['path_logs_train'])}")  # <path_model_id>\\logs\\run.log

    # Define optimizer
    if use_learning_rate_scheduling == True:

        print("Continues WITH using learning rate scheduling")

        num_examples = 2 * ds_train_info.splits['train'].num_examples   # 'mal 2' wg SimCLR
        print("num_examples: ", num_examples, "         //100.000 for cifar10")

        # Jetzt haben wir 291 steps. Diese setzen sich aus 2u + 1s zusammen:
        steps_just_unsupervised = n_epochs * ((num_examples // size_batch) - 1)
        # Da wir jede 2 steps unsupervised jetzt noch 1 step semi-supervised ausführen:
        total_steps = tf.math.floor(3/2 * steps_just_unsupervised)

        print("total_steps:", total_steps, "        //so warmup should be over after step:", math.ceil(total_steps * warmupDuration))

        optimizer = ks.optimizers.Adam(learning_rate=lr_schedule(lr_max=lr_max_ifScheduling, overallSteps=total_steps, warmupDuration=warmupDuration))
        optimizer_head = ks.optimizers.Adam(learning_rate=lr_schedule(lr_max=lr_max_ifScheduling, overallSteps=total_steps, warmupDuration=warmupDuration))

    else:
        print("Continues WITHOUT using learning rate scheduling!")
        optimizer = ks.optimizers.Adam(learning_rate=learning_rate_noScheduling)  #used for both: resnetModel (use_split_model=false) and encoderModel (use_split_model=True)
        optimizer_head = ks.optimizers.Adam(learning_rate=learning_rate_noScheduling)

    # Define checkpoints and checkpoint manager
    # manager automatically handles model reloading if directory contains ckpts

    # Checkpoint für h!     # Oder mit split_model = False Ceckpoint fürs Gesamtmodel g(h(•))
    ckpt = tf.train.Checkpoint(net=model, opt=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=run_paths['path_ckpts_train'],    # <path_model_id>\\ckpts
                                              max_to_keep=2, keep_checkpoint_every_n_hours=None)
    ckpt.restore(ckpt_manager.latest_checkpoint)

    if ckpt_manager.latest_checkpoint:
        logging.info(f"Restored from {ckpt_manager.latest_checkpoint}.")
        epoch_start = int(os.path.basename(ckpt_manager.latest_checkpoint).split('-')[1])+1 #starte bei [letzte_angefangene_epoch + 1]
    else:
        logging.info("Initializing encoder_h from scratch.")
        epoch_start = 0

    # Checkpoint für g!
    ckpt_head = tf.train.Checkpoint(net=model_classifierHead, opt=optimizer)
    ckpt_manager_head = tf.train.CheckpointManager(ckpt_head, directory=run_paths['path_ckpts_projectionhead'],
                                                   max_to_keep=2, keep_checkpoint_every_n_hours=None)
    ckpt_head.restore(ckpt_manager_head.latest_checkpoint)

    if ckpt_manager_head.latest_checkpoint:
        logging.info(f"Restored from {ckpt_manager_head.latest_checkpoint}.")
        epoch_start = int(os.path.basename(ckpt_manager_head.latest_checkpoint).split('-')[1])+1 #starte bei [letzte_angefangene_epoch + 1]
    else:
        logging.info("Initializing projection_head from scratch.")
        epoch_start = 0


    # Define Metrics
    metric_loss_train = ks.metrics.Mean()

    logging.info(f"Training from epoch {epoch_start+1} to {n_epochs}.")
    # use tf variable for epoch passing - so no new trace is triggered
    # if using normal range (instead of tf.range) assign a epoch_tf tensor, otherwise function gets recreated every turn
    epoch_tf = tf.Variable(1, dtype=tf.int32)


    for epoch in range(epoch_start, int(n_epochs)):
        # assign tf variable, graph build doesn't get triggered again
        epoch_tf.assign(epoch)

        logging.info(f"Epoch {epoch + 1}/{n_epochs}: starting training.")

        # Train SimCLR semi-supervised AND unsupervised every epoch

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        for (image, image2, _), (labeled_image, label) in zip(ds_train, train_batches):
            train_step(model, model_head, model_classifierHead, image, image2, labeled_image, label, optimizer,
                       optimizer_head, metric_loss_train, epoch_tf,
                       use_2optimizers=use_2optimizers, batch_size=size_batch, tau=tau,
                       use_lrScheduling=use_learning_rate_scheduling, loss_object=loss_object)

        # Print summary
        if epoch <=0:
            if use_split_model:
                print("Summary des Encoders f(•):")
                model.summary()
                print("Summary des Projection Head g(•):")
                model_head.summary()
            else:
                print("Summary des Gesamtmodels f(•) und g(•):")
                model.summary()
            print("Summary des Projection Head MIT classification Layer:")
            model_classifierHead.summary()

        # Fetch metrics of unsupervised epochs
        logging.info(f"Epoch {epoch + 1}/{n_epochs}: fetching metrics.")
        loss_train_avg = metric_loss_train.result()

        # Log to tensorboard
        with writer.as_default():
            tf.summary.scalar('loss_unsupervisedSimCLR_train_average', loss_train_avg, step=epoch)

        logging.info(f'Epoch {epoch + 1}/{n_epochs}: loss_unsupervisedSimCLR_train_average: {loss_train_avg}')

        # Saving checkpoints
        if epoch % save_period == 0:
            logging.info(f'Saving checkpoint to {run_paths["path_ckpts_train"]}.')  # <path_model_id>\\ckpts    ,z.B. C\\Users\\...\\run_datumTuhrzeit\\ckpts
            ckpt_manager.save(checkpoint_number=epoch)                              # bzw. beim ISS /misc/usrhomes/s1349/experiments/models/run_2020-05-16T09-23-51/ckpts

        if epoch % save_period == 0:
            logging.info(f'Saving checkpoint to {run_paths["path_ckpts_projectionhead"]}.')
            ckpt_manager_head.save(checkpoint_number=epoch)

        # write config after everything has been established
        if epoch <= 0:
            gin_string = gin.operative_config_str()
            logging.info(f'Fetched config parameters: {gin_string}.')
            utils_params.save_gin(run_paths['path_gin'], gin_string)    # <path_model_id>\\config_operative.gin

    return 0



@tf.function
def train_step(model, model_head, model_classifierHead, image, image2, labeled_image, label_ssl, optimizer, optimizer_head,
               metric_loss_train, epoch_tf, use_2optimizers, batch_size, tau, use_lrScheduling, loss_object, gamma=1):

    logging.info(f'Trace indicator - train epoch - eager mode: {tf.executing_eagerly()}.')

    with tf.device('/gpu:*'):
        with tf.GradientTape() as tape:
            ### 1) unsupervised SimCLR loss
            h_i = model(image, training=True)
            z_i = model_head(h_i, training=True)
            h_j = model(image2, training=True)
            z_j = model_head(h_j, training=True)

            z_i = tf.math.l2_normalize(z_i, axis=1)
            z_j = tf.math.l2_normalize(z_j, axis=1)

            l_pos = _dot_simililarity_dim1(z_i, z_j)
            l_pos = tf.reshape(l_pos, (batch_size, 1) )
            l_pos = l_pos / tau

            negatives = tf.concat([z_j, z_i], axis=0)

            loss = 0

            for positives in [z_i, z_j]:
                l_neg = _dot_simililarity_dim2(positives, negatives)

                labels = tf.zeros(batch_size, dtype=tf.int32)

                l_neg = tf.boolean_mask( l_neg,  get_negative_mask(batch_size) )
                l_neg = tf.reshape(l_neg, (batch_size, -1) )
                l_neg = l_neg / tau

                logits = tf.concat([l_pos, l_neg], axis=1)

                loss += tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)(y_pred=logits, y_true=labels)

            loss = loss / (2 * batch_size)

            ### 2) semi-supervised loss
            a = model(labeled_image, training=True)
            b = model_classifierHead(a, training=True)

            loss_ssl = loss_object(label_ssl, b)  # additionaly change all "another_model_head" back to "model_classifierHead"
            # loss = supervised_nt_xent_loss(b, labels, temperature=tau, base_temperature=0.07)

            ### 3) unsupervised and semi-supervised together
            loss_all = loss + gamma * loss_ssl

            tf.summary.scalar('loss', loss_all, step=optimizer.iterations)


        gradients = tape.gradient(loss_all, [model.trainable_variables, model_head.trainable_variables,
                                             model_classifierHead.trainable_variables])        #gradients ist jz liste mit 3 Elementen [0], [1] und [2]

        # use 1 optimizer
        optimizer.apply_gradients(zip(gradients[0], model.trainable_variables))
        optimizer.apply_gradients(zip(gradients[1], model_head.trainable_variables))
        optimizer.apply_gradients(zip(gradients[2], model_classifierHead.trainable_variables))

    # Update metrics
    metric_loss_train.update_state(loss_all)
    if use_lrScheduling:
        tf.print("Training loss for epoch:", epoch_tf + 1, " and step: ", optimizer.iterations, " - ", loss_all,
             "   \tcurrent lr is:", optimizer.learning_rate(tf.cast(optimizer.iterations, tf.float32)),
             output_stream=sys.stdout)
    else:
        tf.print("Training loss for epoch:", epoch_tf + 1, " and step: ", optimizer.iterations, " - ", loss_all,
                 "   \tcurrent lr is:", optimizer.learning_rate,
                 output_stream=sys.stdout)

    return 0