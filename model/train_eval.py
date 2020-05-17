import os
import logging
import tensorflow as tf
import tensorflow.keras as ks
import gin
import sys
from utils import utils_params


#@gin.configurable(blacklist=['model', 'run_paths'])
def train_eval(model,
                   run_paths,
                   n_epochs=1,
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
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory=run_paths['path_ckpts_train'], #C:\Users\Mari\PycharmProjects\experiments\models\run_2020-05-14T19-00-15['path_ckpts_train']
                                              max_to_keep=2, keep_checkpoint_every_n_hours=1)
    ckpt.restore(ckpt_manager.latest_checkpoint)  #Nimmt sich wohl [model_checkpoint_path: "ckpt-55"] aus dem Ordner

    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    return model

