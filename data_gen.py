import numpy as np
import os
import glob
import sys
import random

# Third party 
import keras
import tensorflow as tf
import scipy.io as sio
from keras.utils import to_categorical
import numpy as np
from keras.layers import Input

# Dr.Adrian Dalca's toolbox
sys.path.append('ext/pynd-lib')
sys.path.append('ext/neuron')
sys.path.append('ext/pytools-lib')
sys.path.append('ext/pytools-lib/pytools')
import patchlib
import generators
import dataproc
import layers as nrn_layers

# data_gen.py generates the patches from the imported image. 
# vols_generator_patch: generate the patches from the volumes given and store each patch in consecutive order

def vols_generator_patch (vol_name, num_data, patch_size, stride_patch=32, out = 1):

    # 120 = number of patches for one volume
    vol_patch_only = np.empty([num_data*120,64,64,64,1])
    vol_patch = []
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
            vol_patch.append(temp)
            patch_loc.append(loc_temp)
        elif out == 1:
            for item in patchlib.patch_gen(data_vol,patch_size,stride=stride_patch):
                vol_patch_only[count,:,:,:,0] = item
                count+=1
    if out == 1:
        return vol_patch_only
    elif out == 2:
        return vol_patch, patch_loc

def vol_and_label_generator_patch (vol_name,label_name,trf_vol_model, trf_label_model, resize_model, num_data, patch_size, labels, num_labels=30, stride_patch=32):
    
    # random field deformation augmentation is performed which makes model robust.
    vol_patch = np.empty([120,64,64,64,1]) # patch size is 64,64,64 and 120 patches per volume
    labels_patch = np.empty([120,64,64,64,30]) # 30 label is used
    vol_size = (160, 192, 224) # size of the image
    i = np.random.randint(num_data) # to randomly pick the data
    da_factor = np.random.random(1) * 5 # control the deformation by restrict the max

    rand_phi = np.random.randn(10,12,14,3) * da_factor # play around the initial random array to find best fit for data
    rand_phi = rand_phi[np.newaxis, ...]
    resize_vol = resize_model.predict([rand_phi]) # resize the random field to become smooth


    data_vol =  np.load(vol_name[i])['vol_data'] # load the volume data from the list


    data_vol = data_vol[np.newaxis, ..., np.newaxis]
    data_vol = trf_vol_model.predict([data_vol, resize_vol])[0,...,0] # warp the the image to random field

    data_label =  np.load(label_name[i])['vol_data'] # load the label data
    data_label = generators._relabel(data_label,labels) # relabel
    data_label = to_categorical(data_label,num_labels)  # to one_hot matrix
    data_label = data_label[np.newaxis, ...] 
    data_label = trf_label_model.predict([data_label, resize_vol])[0,...,:] # warp label to random field
    data_label = np.argmax(data_label,axis = -1)

    for idx, item in enumerate(patchlib.patch_gen(data_vol,patch_size,stride=stride_patch)): 
        vol_patch[idx,:,:,:,0] = item
        
    for idx, item in enumerate(patchlib.patch_gen(data_label,patch_size,stride=stride_patch)):
        item = to_categorical(item,num_labels) # change to one-hot representation
        labels_patch[idx,:,:,:,:] = item

    return vol_patch, labels_patch

def example_gen(vol_names):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
    """

    while True:
        dataTrain = np.random.randint(2)
        #dataTrain = 0
        if dataTrain == 0:
            idxes = np.random.randint(len(vol_names))
            X = load_volfile(vol_names[idxes])
            return_vals = [X]

            # also return segmentations
            seg_names = vol_names[idxes].replace('vols', 'asegs')
            seg_names = seg_names.replace('norm', 'aseg')
            X_seg = load_volfile(seg_names)
            return_vals.append(X_seg)

            yield tuple(return_vals)
        else:
            idxes = np.random.randint(len(vol_names))
            vol = load_volfile(vol_names[idxes])
            return_vals = [vol]
            aseg = load_volfile(vol_names[idxes].replace('vol_data', 'labels'))
            return_vals.append(aseg)

            yield tuple(return_vals)

def load_volfile(datafile):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), assume variable names 'vol_data' 
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nibabel' not in sys.modules:
            try :
                import nibabel as nib  
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()
        
    else: # npz
        X = np.load(datafile)['vol_data']

    return X

def re_label(label_input,num_data,labels):
    relabel = []
    for i in range(num_data):
        data_label =  np.load(label_input[i])['vol_data']
        data_label = generators._relabel(data_label,labels)
        relabel.append(data_label)
    return relabel

def linear_label_trf(vol_size, indexing='xy'):
    """
    Simple transform model for nearest-neighbor based transformation
    Note: this is essentially a wrapper for the neuron.utils.transform(..., interp_method='nearest')
    """
    ndims = len(vol_size)

    # linear warp model
    subj_input = Input((*vol_size, 30), name='subj_input')
    trf_input = Input((*vol_size, ndims) , name='trf_input')

    nn_output = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)
    nn_spatial_output = nn_output([subj_input, trf_input])
    return keras.models.Model([subj_input, trf_input], nn_spatial_output)

def linear_vol_trf(vol_size, indexing='xy'):
    """
    Simple transform model for nearest-neighbor based transformation
    Note: this is essentially a wrapper for the neuron.utils.transform(..., interp_method='nearest')
    """
    ndims = len(vol_size)

    # linear warp model
    subj_input = Input((*vol_size, 1), name='subj_input')
    trf_input = Input((*vol_size, ndims) , name='trf_input')

    nn_output = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)
    nn_spatial_output = nn_output([subj_input, trf_input])
    return keras.models.Model([subj_input, trf_input], nn_spatial_output)

def resize_flow(zoom_factor = 16):
    input_shape = (10,12,14)

    flow_input = Input((*input_shape,3))
    resize_output = nrn_layers.Resize(zoom_factor)
    resize_output= resize_output([flow_input])
    
    return keras.models.Model([flow_input], resize_output)

def re_label(label_input,num_data,labels):
    '''
    This function relabels the given label
    input - label_input: input label data(3D)
            num_data: number of data in label_input
            labels: labels of interest
    This function usese generators function from Neuron toolbox
    '''
    relabel = []
    for i in range(num_data):
        data_label =  np.load(label_input[i])['vol_data']
        data_label = generators._relabel(data_label,labels)
        relabel.append(data_label)
    return relabel