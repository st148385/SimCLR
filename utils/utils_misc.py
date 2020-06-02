import matplotlib
matplotlib.use('TkAgg')  # qt may not work on server
from matplotlib import pyplot as plt


import logging

import gin
import tensorflow as tf



def set_loggers(path_log=None, logging_level=0, b_stream=True, b_debug=False):

    # std. logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # tf logger
    logger_tf = tf.get_logger()
    logger_tf.setLevel(logging_level)

    if path_log:
        file_handler = logging.FileHandler(path_log)
        logger.addHandler(file_handler)
        logger_tf.addHandler(file_handler)

    # plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    if b_debug:
        tf.debugging.set_log_device_placement(False)


# def save_dict_to_yaml(d, path_yaml):
#     with open(path_yaml, 'w') as file:
#         # We need to convert the values to float for yaml (it doesn't accept np.array, np.float, )
#         # TODO: check if this is the case for yaml
#         d = {k: float(v) for k, v in d.items()}
#         yaml.dump(d, file, indent=4)

@gin.configurable()
def plot_dataset(ds, dataset_name=' '):
        """Takes ds.take(N) and plots 3*N images. To be exact, it's N instances of augmented_image_A, augmented_image_B, original_Image"""

        for images, images2, labels in ds:
            plt.figure()
            plt.imshow(images[0])
            plt.title("Augmentation 1")
            plt.show()
            plt.figure()
            plt.imshow(images2[0])
            plt.title("Augmentation 2")
            plt.show()
            plt.figure()
            plt.imshow(labels[0])
            plt.title("Original {} Image".format(dataset_name))
            plt.show()

#Von einem Kollegen vom ISS: Nur besten checkpoint speichern (der mit einer besseren eval_accuracy, als alle Vorgänger)

# Save latest eval metrics in a json file in the model directory

#             metrics_path_last = os.path.join(params.path_model_root, "metrics_eval_last.yaml")
#             utils_misc.save_dict_to_yaml(metrics_res, metrics_path_last)
#
#             # If best_eval, best_save_path
#             eval_acc = metrics_res['mean_iou']            #Hier statt 'metrics_res['mean_iou']' dann meine validation accuracy (val_acc) verwenden.
#             if eval_acc >= best_eval_acc:
#                 # Store new best accuracy
#                 logging.info(f'Found new best metric, new: {eval_acc}, old: {best_eval_acc}')
#                 best_eval_acc = eval_acc
#
#                 # Save weights
#                 ckpt.save(os.path.join(params.path_ckpts_eval, 'ckpt'))
#                 logging.info(f'New best model saved')
#
#                 # Save best eval metrics in a yamlm file in the model directory
#                 metrics_path_best = os.path.join(params.path_model_root, "metrics_eval_best.yaml")
#                 utils_misc.save_dict_to_yaml(metrics_res, metrics_path_best)
