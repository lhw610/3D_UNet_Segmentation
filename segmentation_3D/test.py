import os
import glob
import sys
import random
from argparse import ArgumentParser
import scipy.io as sio

# Third party
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model, Model
from keras.optimizers import Adam
from skimage.io import imsave

# my U-Net and data generation functions
import network
import data_generator
from utils import assemble_patches

def inference(dir, data_type, channel, model_name, iters, save_dir, num_classes, gpu_id):

    # read images from given directory
    inference_img_list = glob.glob(dir + '/*.' + data_type)

    # patch size
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

    # save dir
    save = "./inference/" + save_dir
    if not os.path.isdir(save):
        os.mkdir(save)

    with tf.device(gpu):
            model = network.unet(input_size=(64,64,64,1),label_nums=num_classes)
            model.load_weights('/media/lhw610/HD_1T/models/3d_segmentation/' + model_name + '/' + 'weights-' + str(iters) + '.hdf5')
            for idx, inference_img in enumerate(inference_img_list):
                img_patches = data_generator.test_data_gen(inference_img, patch_size, num_classes, channel=channel)
                predict_patch = []
                for patch in img_patches[0]:
                    patch = patch[np.newaxis, ..., np.newaxis].astype('float32')
                    predict_patch.append(np.argmax(model.predict(patch)[0,:,:,:,:], axis=-1))

                full_img = assemble_patches(predict_patch, img_patches[1], img_patches[2])
                imsave(save + "/predicted_img" + str(idx) + ".tiff", full_img.astype('uint8'))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str,required=True,dest="dir", help="File path to the source")
    parser.add_argument("--data_type", type=str,required=True,dest="data_type", help="data type of your image file")
    parser.add_argument("--ch", type=int,required=False,default=2,dest="channel", help="channel of the image if it is multi_channel")
    parser.add_argument("--model_name", type=str,required=True,dest="model_name", help="Name of trained model to import weight from")
    parser.add_argument("--iters", type=int,required=True, dest="iters", help="number of epoch of trained model")
    parser.add_argument("--save_dir", type=str,required=True,dest="save_dir", default=None, help="result saving directory")
    parser.add_argument("--nclasses", type=str,required=False,dest="num_classes", default=2, help="number of labels")
    parser.add_argument("--gpu", type=int,required=False, default = 0, dest="gpu_id", help="gpu id number")
    args = parser.parse_args()
    inference(**vars(args))
