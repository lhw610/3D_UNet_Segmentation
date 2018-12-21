import numpy as np 
import os
import keras.layers
from keras import backend as keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D
from keras.optimizers import Adam


def unet(input_size=None, label_nums=30): #slices = 224
    if input_size is None:
        input_size = (64, 64, 64, 1)
    inputs = Input(input_size)

    conv1 = Conv3D(16, 3, activation = 'relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(inputs)
    conv1 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(conv1)
    
    pool1 = MaxPooling3D(pool_size=(2,2,2), data_format ="channels_last")(conv1)
    conv2 = Conv3D(32, 3, activation ='relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(pool1)
    conv2 = Conv3D(64, 3, activation ='relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(conv2)

    pool2 = MaxPooling3D(pool_size=(2,2,2), data_format ="channels_last")(conv2)
    conv3 = Conv3D(64, 3, activation ='relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(pool2)
    conv3 = Conv3D(128, 3, activation ='relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(conv3)

    pool3 = MaxPooling3D(pool_size=(2,2,2), data_format ="channels_last")(conv3)
    conv4 = Conv3D(128, 3, activation ='relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(256, 3, activation ='relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(conv4)

    pool4 = MaxPooling3D(pool_size=(2,2,2), data_format ="channels_last")(conv4)
    conv5 = Conv3D(256, 3, activation ='relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(pool4)
    conv5 = Conv3D(512, 3, activation ='relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(conv5)

    up1 = UpSampling3D(size=(2,2,2), data_format ="channels_last")(conv5)
    up1 = concatenate([conv4,up1],axis=-1)
    conv6 = Conv3D(256, 3, activation = 'relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(up1)
    conv6 = Conv3D(256, 3, activation = 'relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(conv6)

    up2 = UpSampling3D(size=(2,2,2), data_format ="channels_last")(conv6)
    up2 = concatenate([conv3,up2],axis=-1)
    conv7 = Conv3D(128, 3, activation = 'relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(up2)
    conv7 = Conv3D(128, 3, activation = 'relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(conv7)

    up3 = UpSampling3D(size=(2,2,2), data_format ="channels_last")(conv7)
    up3 = concatenate([conv2,up3],axis=-1)
    conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(up3)
    conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(conv8)

    up4 = UpSampling3D(size=(2,2,2), data_format ="channels_last")(conv8)
    up4 = concatenate([conv1,up4],axis=-1)
    conv9 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(up4)
    conv9 = Conv3D(32, 3, activation = 'relu', padding = 'same', data_format ="channels_last", kernel_initializer='he_normal')(conv9)

    conv10 = Conv3D(label_nums, 1, activation='softmax')(conv9)

    return Model(input=inputs, output=conv10)