import os
import glob
import sys
import random
from argparse import ArgumentParser
import scipy.io as sio

# Third party
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model, Model
from keras.optimizers import Adam

# my U-Net and data generation functions
import network
import data_gen

# Dr. Adrian Dalaca's toolbox
sys.path.append('ext/medipy-lib')
from medipy.metrics import dice
sys.path.append('ext/neuron')
import models

# base data directory
base_data_dir = 'Your Data directory'
validation_vols_data = glob.glob(base_data_dir + 'validation/vol_data/*.npz')
validation_labels_data = glob.glob(base_data_dir + 'validation/labels/*.npz')
validation_vols_data.sort()
validation_labels_data.sort()

# shows the first 5 validation data
for i in range(5):
    print(validation_vols_data[i], validation_labels_data[i])

def test(model_name, iters, gpu_id):
    patch_size = (64,64,64)
    num_labels = 30
    labels = sio.loadmat('labels.mat')['labels'][0]
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    validation_vols, vol_patch_loc = data_gen.vols_generator_patch(validation_vols_data,len(validation_vols_data),patch_size,stride_patch=32,out=2)

    with tf.device(gpu):
            model = network.unet(input_size=(64,64,64,1))
            model.load_weights('models/' + model_name + '/' + 'weights-' + str(iters) + '.hdf5')

    data_temp = np.load(validation_vols_data[0])['vol_data']
    subject_aseg = data_gen.re_label(validation_labels_data,len(validation_labels_data),labels)
    dice_scores = []

    # Make all the patches to one volume then predict by averaging the probability map. Argmax it to the last axis to choose the the label.
    for j in range(len(validation_vols)):
        mask = np.empty(data_temp.shape + (num_labels,))
        for i in range(len(validation_vols[j])):
            pred_temp = model.predict(validation_vols[j][i])
            mask[vol_patch_loc[j][i][0].start:vol_patch_loc[j][i][0].stop,
            vol_patch_loc[j][i][1].start:vol_patch_loc[j][i][1].stop,vol_patch_loc[j][i][2].start:vol_patch_loc[j][i][2].stop,:] += pred_temp[0,:,:,:,:]
        mask /= 4
        pred =  np.argmax(mask, axis = -1)
        vals, _ = dice(pred, subject_aseg[j],nargout=2)
        dice_scores.append(np.mean(vals))
    print("dice:",np.mean(dice_scores),model_name,iters)

if __name__ == "__main__":
        test(sys.argv[1], sys.argv[2], sys.argv[3])
