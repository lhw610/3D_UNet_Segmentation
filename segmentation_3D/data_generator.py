import numpy as np
import os
import glob
import sys
import random

# Third party 
import tensorflow as tf
from keras.utils import to_categorical

from skimage.io import imread, imsave

import utils

import pdb
from skimage.io import imsave

def train_data_gen(train, target, patch_size, num_classes, trf_vol_model, trf_label_model,channel=None):
    '''
    This function generates random patches and perform spatial data augmentation
    '''
    while True:
        # randomly choose one image
        train_num = len(train)
        selected = np.random.randint(low = 0, high = train_num)
        source_img = imread(train[selected])[:,:,:,channel-1]
        target_img = imread(target[selected])

        # intensity normalization
        source_img = utils.intensity_norm(source_img, [3.5, 15])

        # extract patches
        patches = utils.rand_patch_gen(source_img, target_img, patch_size, num_classes, trf_vol_model, trf_label_model)

        yield (patches.__next__())


def test_data_gen(img, patch_size, num_classes,channel):
    '''
    This function returns pathes with given size
    '''
    img = imread(img)[:,:,:,channel-1]
    img = utils.intensity_norm(img, [3.5, 15])
    img_size = img.shape

    # decide padding region
    z_pad = patch_size[0] - (img_size[0] % patch_size[0])
    y_pad = patch_size[1] - (img_size[1] % patch_size[1]) 
    x_pad = patch_size[2] - (img_size[2] % patch_size[2]) 

    template = np.zeros((img_size[0]+z_pad, img_size[1]+y_pad, img_size[2]+x_pad))
    template[0:img_size[0], 0:img_size[1], 0:img_size[2]] = img

    patch_list = []

    for x_patch in range(0,template.shape[2],patch_size[2]):
        for y_patch in range(0,template.shape[1],patch_size[1]):
            for z_patch in range(0,template.shape[0],patch_size[0]):
                patch_list.append(template[z_patch:(z_patch+patch_size[0]),y_patch:(y_patch+patch_size[1]), x_patch:(x_patch+patch_size[2])])

    return (patch_list, (z_pad, y_pad, x_pad), img_size)