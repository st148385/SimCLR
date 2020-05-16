import os
import logging
import tensorflow as tf
import tensorflow.keras as ks
import gin
import sys
from utils import utils_params


@gin.configurable(blacklist=['model', 'run_paths'])
def train_eval(model,
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
    restoration = ckpt.restore(ckpt_manager.latest_checkpoint)

    if ckpt_manager.latest_checkpoint:
        logging.info(f"Restored from {ckpt_manager.latest_checkpoint}.")
    else:
        logging.info("Initializing from scratch.")

    return restoration