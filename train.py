import logging
from model import input_fn, model_fn
from model.train_representations import train
from utils import utils_params, utils_misc, utils_devices
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

##########
def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label)        #plt.title(label.numpy().decode('utf-8'))
    plt.axis('off')
##########

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

    # and evalutation pipeline:
    # ds_eval, ds_eval_info = input_fn.gen_pipeline_eval()
    # (Evaluation is maybe in another script, cause this script will only train the model in a unsupervised way. No eval needed)




    print("print ds_train:\n", ds_train, "\n")   #<PrefetchDataset shapes: ((128, 224, 224, 3), (128, 224, 224, 3), (128, 224, 224, 3)), types: (tf.float32, tf.float32, tf.float32)>
    #print("print d_train_info:\n", ds_train_info, "\n")                    #tf_flowers
    #print("print np.shape(ds_train):\n",np.shape(ds_train),"\n")            #()



    im_size=224 #Vorübergehend zum Plotten
    orig_im_size=219
    for image1, image2, label in ds_train.take(1):                                      #PrefetchDataset.take(N) nimmt also N image1-, image2- und label-Batches aus ds_train
            print("shape(image1):", np.shape(image1),  "\n\n\n")        #(128, im_size, im_size, 3)
            print("shape(image2):", np.shape(image2),  "\n\n\n")        #(128, im_size, im_size, 3)
            print("shape(original):", np.shape(label), "\n\n\n")        #(128, im_size, im_size, 3)


            # show(tf.reshape(image1[:1], (im_size,im_size,3) ), label='x_i_1 von Batch1')        #Erster Eintrag des EINEN Batch von ds_train.take(1) -> x_i
            # show(tf.reshape(image1[1:2], (im_size, im_size, 3)), label='x_i_2 von Batch 1')     #x_i des zweiten Images von Batch1
            #
            # show(tf.reshape(image2[:1], (im_size, im_size, 3)), label='x_j_1 von Batch1')       #Auch erster Eintrag, aber versch. augmentiert -> x_j
            # show(tf.reshape(image2[1:2], (im_size, im_size, 3)), label='x_j_2 von batch1')      #x_j des zweiten Images von Batch1
            #
            # show( tf.reshape( label[:1], (orig_im_size,orig_im_size,3) ) , label='Original x_1')#Originale Version des ersten Eintrages der Batch -> x
            # show(tf.reshape(label[1:2], (orig_im_size, orig_im_size, 3)), label='Original x_2') #zweites Image von Batch1



    # Define model
    #model = model_fn.gen_Model()
    encoder_f = model_fn.gen_encoderModel()
    projectionhead_g = model_fn.gen_headModel()
    gesamtmodel_f_g = model_fn.gen_model()

    train(encoder_f,
          projectionhead_g,
          gesamtmodel_f_g,
          ds_train,
          ds_train_info,
          run_paths)


#main()
if __name__ == '__main__':
    device = '0'
    path_model_id = ''
    #path_model_id = 'C:\\Users\\Mari\\PycharmProjects\\experiments\\models\\run_2020-05-14T19-00-15'   # only to use if starting from existing model



    # gin config files
    config_names = ['config.gin', 'architecture.gin']



    # start training
    set_up_train(path_model_id=path_model_id, device=device, config_names=config_names)



