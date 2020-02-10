import os
import numpy as np
import glob
import sys
import random
from argparse import ArgumentParser
import scipy.io as sio

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import keras
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy

from keras_tqdm import TQDMCallback

import network
import data_generator
import utils

def train(source,
          target,
          data_type,
          channel,
          save_name, 
          gpu_id, 
          iters,
          load_model,
          save_iters,
          num_labels,
          batch,
          ):
    
    # read images from given directory
    train_img_list = glob.glob(source + '/*.' + data_type)
    target_img_list = glob.glob(target + '/*.' + data_type)

    train_img_list.sort()
    target_img_list.sort()

    # patch_size
    patch_size = (64,64,64)

    # GPU setting
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpu = '/GPU:' + str(gpu_id) 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # GPU memory limit block #############################
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)
    ######################################################

    # check if model save path exists
    model_dir = './model/' + save_name
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    trf_label_model = utils.linear_label_trf(patch_size, num_labels, indexing='ij')
    trf_vol_model = utils.linear_vol_trf(patch_size, indexing='ij')

    # trainset generator
    train_data = data_generator.train_data_gen(train_img_list, target_img_list, patch_size, num_labels, trf_vol_model, trf_label_model,channel=channel)


    with tf.device(gpu):
        # model configuration
        model = network.unet(input_size=(patch_size[0],patch_size[1],patch_size[2],1), label_nums=num_labels) 
        model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=[utils.dice_coef])

        # load weights of model when continue training
        if load_model is not None:
            print('loading', load_model)
            model.load_weights(load_model)

        # train the model
        checkpoint = keras.callbacks.ModelCheckpoint('/media/lhw610/HD_1T/models/3d_segmentation/'+save_name+'/'+"weights-{epoch:02d}.hdf5", monitor='loss_1',
                        verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=save_iters)
        callbacks_list = [checkpoint, TQDMCallback()]
        model.fit(train_data,
                  steps_per_epoch= batch, 
                  epochs=iters,
                  verbose=0,
                  callbacks=callbacks_list)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source", type=str,required=True,dest="source", help="File path to the source")
    parser.add_argument("--target", type=str,required=True,dest="target", help="File path to the target")
    parser.add_argument("--data_type", type=str,required=False, default= "tiff", dest="data_type", help="data type of your image file")
    parser.add_argument("--ch", type=int,required=False,default=2,dest="channel", help="channel of the image if it is multi_channel")
    parser.add_argument("--save_name", type=str,required=True,dest="save_name", help="Name of training model for saving")
    parser.add_argument("--gpu", type=int,required=False, default = 0, dest="gpu_id", help="gpu id number")
    parser.add_argument("--iters", type=int,required=True, dest="iters", help="number of epoch")
    parser.add_argument("--load", type=str,required=False,dest="load_model", default=None, help="model to continue from")
    parser.add_argument("--save_iters", type=int,required=True, dest="save_iters", help="saving iters")
    parser.add_argument("--nclasses", type=int, default=2, dest="num_labels", help="number of labels")
    parser.add_argument("--batch", type=int, default=20, dest="batch", help="batch size")

    args = parser.parse_args()
    train(**vars(args))
