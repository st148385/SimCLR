import logging
from model import input_fn, model_fn
# from model import train_SSL            #Just take one of train_SSL or train_SSL_verzahnt
from model import train_SSL_verzahnt
from utils import utils_params, utils_misc, utils_devices


def set_up_train(path_model_id = '', device='0', config_names=['config.gin']):

    # generate folder structures
    run_paths = utils_params.gen_run_folder(path_model_id=path_model_id)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # config
    utils_params.inject_gin(config_names, path_model_id=path_model_id)

    # set device params
    utils_devices.set_devices(device)

    # generate training pipeline:
    ds_train, ds_train_info = input_fn.gen_pipeline_train()

    # generate semi-supervised pipeline with 15% of images
    #train_batches, test_batches = input_fn.gen_pipeline_ssl_eval('cifar10', tfds_path='/data/public/tensorflow_datasets', BATCH_SIZE=64, useNpercentOfCifar10=15)
    train_batches, test_batches = input_fn.gen_pipeline_ssl_eval('cifar10')


    # Define model
    encoder_h = model_fn.gen_model()
    projectionhead_g = model_fn.gen_headModel()
    gesamtmodel_h_g = model_fn.gen_fullModel()
    projectionhead_g_with_classifier = model_fn.gen_classifierHead()
    #kurzer test:
    another_projectionhead_g = model_fn.gen_headModel()

    # train_SSL.train(encoder_h,
    #       projectionhead_g,
    #       projectionhead_g_with_classifier,
    #       gesamtmodel_h_g,
    #       ds_train, ds_train_info,      # unlabeled dataset
    #       train_batches, test_batches,  # part of fully labeled dataset
    #       run_paths)

    # Also remove 'from model import train_SSL_verzahnt'
    train_SSL_verzahnt.train(encoder_h,
                             projectionhead_g,
                             projectionhead_g_with_classifier,
                             gesamtmodel_h_g,
                             ds_train, ds_train_info,  # unlabeled dataset
                             train_batches, test_batches,  # part of fully labeled dataset
                             run_paths)


#main()
if __name__ == '__main__':
    device = '0'
    path_model_id = ''
    #path_model_id = 'C:\\Users\\Mari\\PycharmProjects\\experiments\\models\\run_2020-05-14T19-00-15'   # only to use if starting from existing model

    # gin config files
    config_names = ['ssl_training_config.gin', 'architecture.gin']

    # start training
    set_up_train(path_model_id=path_model_id, device=device, config_names=config_names)

