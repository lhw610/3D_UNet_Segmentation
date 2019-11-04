import numpy as np
import os
import glob
import sys
import random

# Third party 
import tensorflow as tf
import scipy.io as sio
from keras.utils import to_categorical

# Dr.Adrian Dalca's toolbox
sys.path.append('ext/pynd-lib')
sys.path.append('ext/neuron')
sys.path.append('ext/pytools')
import patchlib
import generators
import dataproc
 
'''
data_gen.py generates the patches from the imported image. 

vols_generator_patch: generate the patches from the volumes given and store each patch in consecutive order
input - vol_name: list(files)
        num_data: int(number of data you want to train)
        patch_size: patch size, 3 dim
        stride_patch: overlapping region
        out: out = 1 returns the list of patches in consecutive order.
             out = 2 returns the list of patches and the corresponding location. Unlike when out = 1,
             patches were grouped with corresponding data.
             e.g. if num_data = 5, len(vol_patch2) = 5. len(vol_patch2[0]) = number of patches for first data

label_generator_patch: generate the patches from the labels given and store each patch in consecutive order
input - label_name: list(files)
        num_data: int(number of label data you want to train)
        patch_size: patch size, 3 dim
        stride_patch: overlapping region
        label: list of labels that you would like to use for relableling
        out: out = 1 returns the list of label patches in consecutive order.
             out = 2 returns the list of patches and the corresponding location. Unlike when out = 1
             patches were grouped with corresponding data.
             e.g. if num_data = 5, len(vol_patch2) = 5. len(vol_patch2[0]) = number of patches for first data
        
        return: list of tensors. tensor.shape = 1 + dimension of patch + number of label

'''

def vols_generator_patch (vol_name, num_data, patch_size, stride_patch=32, out = 1):

    # 120 = number of patches for one volume
    vol_patch = np.empty([num_data*120,64,64,64,1])
    vol_patch2 = []
    patch_loc = []
    count = 0 # count the batch size for the network

    for i in range(num_data):
        data_vol =  np.load(vol_name[i])['vol_data'] # load the volume data from the list
        print("volume data",i,":",vol_name[i]) # print the volume data used for training
        loc_temp = []
        temp = []
        if out == 2: 
            # generate the patch and store them in a list
            for item, loc in patchlib.patch_gen(data_vol,patch_size,stride=stride_patch, nargout = out):
                item = np.reshape(item, (1,) + item.shape + (1,))
                temp.append(item)
                loc_temp.append(loc)
            vol_patch2.append(temp)
            patch_loc.append(loc_temp)
        elif out == 1:
            for item in patchlib.patch_gen(data_vol,patch_size,stride=stride_patch):
                # vol_patch = [batch size, (dimension), channel]
                vol_patch[count,:,:,:,0] = item
                count+=1
    if out == 1:
        return vol_patch
    elif out == 2:
        return vol_patch2, patch_loc

def label_generator_patch (label_name, num_data, patch_size, labels, num_labels, stride_patch=32, out = 1):

    # 120 = number of patches for one volume, 30 = number of labels of interest
    labels_patch = np.empty([num_data*120,64,64,64,30])
    labels_patch2 = []
    patch_loc = []
    count = 0 # count the batch size for the network
    for i in range(num_data):
        data_label =  np.load(label_name[i])['vol_data']    
        print("label data",i,":",label_name[i])
        data_label = generators._relabel(data_label,labels)
        temp = []
        loc_temp = []
        if out == 2:
            # generate the patch and store them in a list
            for item, loc in patchlib.patch_gen(data_label,patch_size,stride=stride_patch, nargout = out):
                item = to_categorical(item,num_labels) # change to one-hot representation
                item = np.reshape(item, (1,) + item.shape)
                temp.append(item)
                loc_temp.append(loc)
            labels_patch2.append(temp)
            patch_loc.append(loc_temp)
        elif out == 1:
            for item in patchlib.patch_gen(data_label,patch_size,stride=stride_patch):
                item = to_categorical(item,num_labels) # change to one-hot representation
                # vol_patch = [batch size, (dimension), channel(30 as default)]
                labels_patch[count,:,:,:,:] = item
                count+=1
    if out == 1:
        return labels_patch
    elif out == 2:
        return labels_patch2, patch_loc

'''
This function relabels the given label
input - label_input: input label data(3D)
        num_data: number of data in label_input
        labels: labels of interest
This function usese generators function from Neuron toolbox
'''

def re_label(label_input,num_data,labels):
    relabel = []
    for i in range(num_data):
        data_label =  np.load(label_input[i])['vol_data']
        data_label = generators._relabel(data_label,labels)
        relabel.append(data_label)
    return relabel
