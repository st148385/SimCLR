import logging
from model import input_fn, model_fn, train_eval
from utils import utils_params, utils_misc, utils_devices
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub



def evaluation_train(path_model_id = '', device='0', config_names=['config.gin']):

    # generate folder structures
    run_paths = utils_params.gen_run_folder(path_model_id=path_model_id)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # config
    utils_params.inject_gin(config_names, path_model_id=path_model_id)

    # set device params
    utils_devices.set_devices(device)

    # evaluation pipeline:
    train_batches, validation_batches = input_fn.gen_pipeline_eval()

    # Get already trained SimCLR model or empty model for Bounds
    model = model_fn.gen_resnet50()
    restored_model = train_eval.load_checkpoint_weights(model=model, run_paths=run_paths)

    # Train the dense layer together with frozen SimCLR model
    train_eval.custom_train_evaluation_network(simclr_encoder_h=restored_model, train_batches=train_batches,
                                               validation_batches=validation_batches, run_paths=run_paths)




#main()
if __name__ == '__main__':
    device = '0'
    path_model_id = 'C:\\Users\\Mari\\PycharmProjects\\experiments\\models\\run_2020-05-16T09-23-51'
    path_model_id = ''

    # gin config files
    config_names = ['config_eval.gin', 'architecture.gin']

    # start training
    evaluation_train(path_model_id=path_model_id, device=device, config_names=config_names)