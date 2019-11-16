import os
import glob
import sys
import random
from argparse import ArgumentParser
import scipy.io as sio

import keras
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model, Model
from keras.optimizers import Adam

import network
# import networks
import data_gen
import losses
import random

sys.path.append('ext/neuron')
import metrics
import models 
from metrics import Dice

# base data directory
base_data_dir = 'Your Data directory'
# I use npz file
train_vols_data = glob.glob(base_data_dir + '/*.npz')
train_labels_data = glob.glob(base_data_dir + '/*.npz')

# Sort the volume and labels to match their names
train_vols_data.sort()
train_labels_data.sort()
    
def train(save_name, gpu_id, num_data,iters,load,checkpoint,num_labels):
    
    # Anatomical Label to evaluate
    labels = sio.loadmat('labels.mat')['labels'][0]

    # Patch size and stride 
    patch_size = [64,64,64]
    stride = 32
    vol_size = (160, 192, 224)

    # Generates the list of random patches for training volume and labels


    gpu = '/gpu:' + str(gpu_id) 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    model_dir = 'models/' + save_name
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    trf_label_model = data_gen.linear_label_trf(vol_size, indexing='ij')
    trf_vol_model = data_gen.linear_vol_trf(vol_size, indexing='ij')
    resize_model = data_gen.resize_flow(zoom_factor = 16)

    with tf.device(gpu):
        # Use dice score for the metric
        # adjust input size based on size of data
        # loss_1 is dice score
        model = network.unet(input_size=(64,64,64,1)) 
        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=[Dice(nb_labels=num_labels).loss])

        # load weights of model when continue training
        if load_model is not None:
            print('loading', load_model)
            model.load_weights(load_model)

        train_data = data_gen.example_gen(train_vols_data)
        initial_epoch = 0
        for step in range(iters):
            train_vols, train_labels = data_gen.vol_and_label_generator_patch(vol_gen = train_data, 
                                                                            trf_vol_model=trf_vol_model,
                                                                            trf_label_model=trf_label_model,
                                                                            resize_model = resize_model, 
                                                                            patch_size = patch_size,
                                                                            labels = labels, 
                                                                            stride_patch = stride)

            train_loss = model.fit(x=train_vols, y=train_labels, epochs=1, batch_size=1, shuffle=True, initial_epoch = initial_epoch,verbose=0)

        if(step % save_iter == 0):
            model.save(model_dir + '/' + str(step) + '.h5')
        if not isinstance(train_loss, list):
            train_loss = [train_loss.history['loss'],train_loss.history['loss_1']]
        # if not isinstance(train_loss, list):
        #     train_loss = [train_loss]

        print(step, 1, train_loss)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_name", type=str,required=True,dest="save_name", help="Name of training model for saving")
    parser.add_argument("--gpu", type=int,required=True, dest="gpu_id", help="gpu id number")
    parser.add_argument("--nb_data", type=int,required=True, dest="num_data", help="number of data will be trained")
    parser.add_argument("--iters", type=int,required=True, dest="iters", help="number of epoch")
    parser.add_argument("--load", type=str,required=False,dest="load_model", default=None, help="model to continue from")
    parser.add_argument("--save_iters", type=int,required=True, dest="checkpoint", help="checkpoint iters")
    parser.add_argument("--num_labels", type=int, default=30, dest="num_labels", help="number of interest labels")

    args = parser.parse_args()
    train(**vars(args))
