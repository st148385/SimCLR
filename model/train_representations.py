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


def anotherLossImplementation(a, b, tau):
    a=tf.reshape(a, (-1,1))
    b=tf.reshape(b, (-1,1))
    a_norm = tf.reshape(tf.math.l2_normalize(a, axis=1), shape=(-1, 1))
    a_hat = tf.math.divide(a, a_norm)                               # â = a / |a|
    b_norm = tf.reshape(tensor=tf.math.l2_normalize(b, axis=1), shape=(-1, 1))
    b_hat = tf.math.divide(b, b_norm)
    a_hat_b_hat = tf.concat([a_hat, b_hat], axis=0)                 #All augmented images i and all images j in one vector => (BS*128*2, 1)
    a_hat_b_hat_transpose = tf.transpose(a_hat_b_hat)               #(1, BS*128*2)
    b_hat_a_hat = tf.concat([b_hat, a_hat], axis=0)
    sim = tf.matmul(a_hat_b_hat, a_hat_b_hat_transpose)             #(BS*128*2, BS*128*2)
    sim_over_tau = tf.math.divide(sim, tau)
    exp_sim_over_tau = tf.exp(sim_over_tau)                         #(BS*128*2, BS*128*2)
    exp_sim_over_tau_diag = tf.linalg.diag_part(exp_sim_over_tau)   #(BS*128*2,) This is the sim with k=i, so sim(z_i,z_i)
    sum_of_rows = tf.reduce_sum(exp_sim_over_tau, axis=1)           #sum of all elements in every one line -> (BS*128*2,)
    denominators = sum_of_rows - exp_sim_over_tau_diag              #Remove the e^(sim(zi,zi)/tau) from the sum -> Denominator/Nenner ready (4096,)

    cosine_sim = tf.keras.losses.CosineSimilarity()(a_hat_b_hat, b_hat_a_hat)   #sim( [concat(â+ĉ),1] , [concat(ĉ+â),1] )  ->  <(N,1),(1,N)>  ->  scalar
    cosine_sim_over_tau = tf.math.divide(cosine_sim, tau)
    numerators = tf.exp(cosine_sim_over_tau)                        #Numerator/Zähler ready (scalar)

    numerator_over_denominator = tf.math.divide(numerators, denominators)   #Last steps are: mean(-log(numerator/denominator))
    negativelog_num_over_den = -tf.math.log(numerator_over_denominator)
    return tf.math.reduce_mean(negativelog_num_over_den)


@gin.configurable(blacklist=['model','model_head','model_gesamt','ds_train', 'ds_train_info', 'run_paths'])     #Eine Variable in der blacklist kann über die config.gin KEINEN Wert erhalten.
def train(model, model_head, model_gesamt,
          ds_train,
          ds_train_info,
          run_paths,
          n_epochs=10,
          learning_rate_noScheduling=0.001,
          lr_max_ifScheduling=0.001,
          save_period=1,
          size_batch=128,
          tau=0.5,
          use_2optimizers=True,
          use_split_model=True,
          use_learning_rate_scheduling=True,
          warmupDuration=0.1
          ):
    """Executes the Training Loop of SimCLR. So it trains the simclr's encoder h(•) and projection head g(•).
    Also tries to load ckpts of started trainings from run_paths and saves trained checkpoints to run_paths.

    Pass both parts of split model and the overall model (all 3 models), so parameter "use_split_model" can be used.

    When using "use_split_model=True", you train two seperate models. Therefore you can now decide, if those seperate
    models also use two seperate optimizers via "use_2optimizers".
    When using "use_learning_rate_scheduling=True" "warmupDuration" (percentages in decimal form) will decide, how many
    steps the linear warmup is executed compared to the steps of cosine decay.
    """

    # Use model_gesamt as model, if use_split_model==False was chosen. Now there's only one model g(h(•)).
    if use_split_model == False:
        model = model_gesamt

    # Generate summary writer
    writer = tf.summary.create_file_writer(os.path.dirname(run_paths['path_logs_train']))
    logging.info(f"Saving log to {os.path.dirname(run_paths['path_logs_train'])}")  # <path_model_id>\\logs\\run.log

    # Define optimizer
    if use_learning_rate_scheduling == True:

        print("Continues WITH using learning rate scheduling")

        num_examples = 2 * ds_train_info.splits['train'].num_examples   # 'mal 2' wg SimCLR
        print("num_examples: ", num_examples, "         //100.000 for cifar10")

        total_steps = n_epochs * ( (num_examples // size_batch) - 1 )
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
    ckpt_head = tf.train.Checkpoint(net=model_head, opt=optimizer_head)
    ckpt_manager_head = tf.train.CheckpointManager(ckpt_head, directory=run_paths['path_ckpts_projectionhead'],    # <path_model_id>\\ckpts\\projectionhead
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

        # Train
        for image, image2, _ in ds_train:
            # Train on batch
            if use_split_model == True:
                train_step(model, model_head, image, image2, optimizer, optimizer_head, metric_loss_train, epoch_tf,
                           use_2optimizers=use_2optimizers, batch_size=size_batch, tau=tau,
                           use_lrScheduling=use_learning_rate_scheduling)
            else:
                train_step_just1model(model, image, image2, optimizer, metric_loss_train, epoch_tf, batch_size=size_batch, tau=tau, use_lrScheduling=use_learning_rate_scheduling)
        # Print summary
        if epoch <=0:
            if use_split_model:
                print("Summary des Encoders f(•):")
                model.summary()
                print("Summary des Projection Head g(•)")
                model_head.summary()
            else:
                print("Summary des Gesamtmodels f(•) und g(•):")
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
def train_step(model, model_head, image, image2, optimizer, optimizer_head, metric_loss_train, epoch_tf, use_2optimizers, batch_size, tau, use_lrScheduling):
    logging.info(f'Trace indicator - train epoch - eager mode: {tf.executing_eagerly()}.')

    with tf.device('/gpu:*'):
        with tf.GradientTape() as tape:
            h_i = model(image, training=True)
            z_i = model_head(h_i, training=True)
            h_j = model(image2, training=True)
            z_j = model_head(h_j, training=True)

            ###
            # loss_obv = 0
            # loss_obv = anotherLossImplementation(z_i, z_j, tau)
            # print("Loss with 'more obvious' implementation:", loss_obv)
            ###

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

            # assert l_pos.shape == (batch_size, 1), "l_pos shape ist falsch!" + str(l_pos.shape)  # [N,1]

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

                # Mask to remove positive examples from the batch of negative samples
                l_neg = tf.boolean_mask( l_neg,  get_negative_mask(batch_size) )        #negative_mask = get_negative_mask(batch_size) # alle elemente der diagnole vervwerfen

                #print("l_neg:\n", l_neg.shape)          #(32512,)       #127*256=32512, bzw. 128*254=32512

                l_neg = tf.reshape(l_neg, (batch_size, -1) )

                #Error: reshape (32512,) -> (128,1). Stattdessen jetzt (32512,) -> (128,32512/128)=(128,254)

                #print("l_neg:\n", l_neg.shape)     #(128,254)


                l_neg = l_neg / tau

                # assert l_neg.shape == (batch_size, 2 * (batch_size - 1)), "Shape of negatives not expected." + str(l_neg.shape)

                #print("l_pos:\n", l_pos.shape)         #(128,1)        (2 Images der 2N=2Batch_Size Images bilden ein positive pair)
                #print("l_neg:\n", l_neg.shape)         #(128,254)      (Die verbleibenden 2N-2=254 Images liefern die negatives)

                logits = tf.concat([l_pos, l_neg], axis=1)  # [N,K+1]   #"logits": "This Tensor is the quantity that is being mapped to probabilities by the Softmax"

                #print("logits:\n", logits)     #(128,255)
                ###Shape: logits=(128,2)

                loss += tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)(y_pred=logits, y_true=labels)

                #print("loss:\n", loss.shape)   #()   ->Skalar


            loss = loss / (2 * batch_size)

            tf.summary.scalar('loss', loss, step=optimizer.iterations)


        #print(model.trainable_variables.shape)         #AttributeError: 'list' object has no attribute 'shape'

        gradients = tape.gradient(loss, [model.trainable_variables, model_head.trainable_variables])        #gradients ist jz liste mit 2 Elementen [0] und [1]

        if use_2optimizers == True:
            optimizer.apply_gradients(zip(gradients[0], model.trainable_variables))
            optimizer_head.apply_gradients(zip(gradients[1], model_head.trainable_variables))
        else:
            optimizer.apply_gradients(zip(gradients[0], model.trainable_variables))
            optimizer.apply_gradients(zip(gradients[1], model_head.trainable_variables))

        #gradients = tape.gradient(loss, model_head.trainable_variables)
        #optimizer.apply_gradients(zip(gradients, model_head.trainable_variables))

    # Update metrics
    metric_loss_train.update_state(loss)
    if use_lrScheduling:
        tf.print("Training loss for epoch:", epoch_tf + 1, " and step: ", optimizer.iterations, " - ", loss,
             "   \tcurrent lr is:", optimizer.learning_rate(tf.cast(optimizer.iterations, tf.float32)),
             output_stream=sys.stdout)
    else:
        tf.print("Training loss for epoch:", epoch_tf + 1, " and step: ", optimizer.iterations, " - ", loss,
                 "   \tcurrent lr is:", optimizer.learning_rate,
                 output_stream=sys.stdout)

    return 0



@tf.function
def train_step_just1model(model, image, image2, optimizer, metric_loss_train, epoch_tf, batch_size, tau, use_lrScheduling):
    logging.info(f'Trace indicator - train epoch - eager mode: {tf.executing_eagerly()}.')


    with tf.device('/gpu:*'):
        with tf.GradientTape() as tape:
            h_i, z_i = model(image, training=True)
            h_j, z_j = model(image2, training=True)


            # normalize projection feature vectors
            z_i = tf.math.l2_normalize(z_i, axis=1)
            z_j = tf.math.l2_normalize(z_j, axis=1)


            l_pos = _dot_simililarity_dim1(z_i, z_j)    #l_pos = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2)), d.h. shape(z_i) wird (•,1,•) und shape(z_j) wird (•,•,1)


            l_pos = tf.reshape(l_pos, (batch_size, 1) ) #l_pos erhält shape=(128,1) -> column vector


            l_pos = l_pos / tau


            negatives = tf.concat([z_j, z_i], axis=0)   #concat: Wenn z_j shape (a,b) und z_i shape (a,b) haben, hat negatives shape (a+a,b)
                                                        #Mit axis=1 hätte im selben Beispiel negatives die shape (a,b+b)

            loss = 0

            for positives in [z_i, z_j]:

                l_neg = _dot_simililarity_dim2(positives, negatives)    #l_neg = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)


                labels = tf.zeros(batch_size, dtype=tf.int32)

                # Mask to remove positive examples from the batch of negative samples
                l_neg = tf.boolean_mask( l_neg,  get_negative_mask(batch_size) )        #negative_mask = get_negative_mask(batch_size) # alle elemente der diagnole vervwerfen


                l_neg = tf.reshape(l_neg, (batch_size, -1) )


                l_neg = l_neg / tau


                logits = tf.concat([l_pos, l_neg], axis=1)  # [N,K+1]   #"logits": "This Tensor is the quantity that is being mapped to probabilities by the Softmax"


                loss += tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)(y_pred=logits, y_true=labels)


            loss = loss / (2 * batch_size)
            tf.summary.scalar('loss', loss, step=optimizer.iterations)


        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    # Update metrics
    metric_loss_train.update_state(loss)
    if use_lrScheduling:
        tf.print("Training loss for epoch:", epoch_tf + 1, " and step: ", optimizer.iterations, " - ", loss,
             "   \tcurrent lr is:", optimizer.learning_rate(tf.cast(optimizer.iterations, tf.float32)),
             output_stream=sys.stdout)
    else:
        tf.print("Training loss for epoch:", epoch_tf + 1, " and step: ", optimizer.iterations, " - ", loss,
                 "   \tcurrent lr is:", optimizer.learning_rate,
                 output_stream=sys.stdout)
    return 0


