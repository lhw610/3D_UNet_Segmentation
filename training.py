import os
import glob
import sys
import random
from argparse import ArgumentParser
import scipy.io as sio

# import thrid party toolboxs
import keras
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model, Model
from keras.optimizers import Adam

# my U-Net and data generation functions
import network
import data_gen

# Dr.Adrian Dalca's toolbox 
sys.path.append('ext/neuron')
import metrics 
from metrics import Dice

# base data directory
base_data_dir = 'Your Data directory'
# I use npz file
train_vols_data = glob.glob(base_data_dir + '/*.npz')
train_labels_data = glob.glob(base_data_dir + '/*.npz')

# Sort the volume and labels to match their names
train_vols_data.sort()
train_labels_data.sort()
    
def train(save_name, gpu_id, num_data,iters,checkpoint,num_labels):
    
    # Anatomical Label to evaluate
    labels = sio.loadmat('labels.mat')['labels'][0]

    # Patch size and stride 
    patch_size = [64,64,64]
    stride = 32

    # Generates the list of patches for training volume and labels
    train_vols = data_gen.vols_generator_patch(vol_name = train_vols_data, num_data = num_data ,patch_size = patch_size, 
                                                stride_patch = stride, out=1)
    train_labels = data_gen.label_generator_patch(label_name = train_labels_data, num_data = num_data ,patch_size = patch_size,
                                                labels = labels, num_labels=num_labels, stride_patch = stride,out=1)
    print(train_vols.shape, train_labels.shape)

    model_dir = 'models/' + save_name
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    gpu = '/gpu:' + str(gpu_id) 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    with tf.device(gpu):
        # Use dice score for the metric
        # adjust input size based on size of data
        # loss_1 is dice score
        model = network.unet(input_size=(64,64,64,1)) 
        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=[Dice(nb_labels=num_labels).loss])
        # load weights of model
        # model.load_weights('models/unet_5/weights-50.hdf5')

        # saving the model for every checkpoint given
        checkpoint = keras.callbacks.ModelCheckpoint('models/'+save_name+'/'+"weights-{epoch:02d}.hdf5", monitor='loss_1',
                        verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=checkpoint)
        callbacks_list = [checkpoint]
        model.fit(x=train_vols, y=train_labels, epochs=iters, batch_size=1, shuffle=True, callbacks=callbacks_list)
 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_name", type=str,required=True,dest="save_name", help="Name of training model for saving")
    parser.add_argument("--gpu", type=int,required=True, dest="gpu_id", help="gpu id number")
    parser.add_argument("--nb_data", type=int,required=True, dest="num_data", help="number of data will be trained")
    parser.add_argument("--iters", type=int,required=True, dest="iters", help="number of epoch")
    parser.add_argument("--save_iters", type=int,required=True, dest="checkpoint", help="checkpoint iters")
    parser.add_argument("--num_labels", type=int, default=30, dest="num_labels", help="number of interest labels")

    args = parser.parse_args()
    train(**vars(args))
