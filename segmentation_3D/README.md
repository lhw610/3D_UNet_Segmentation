# 3D Segmentation
This is an implementation of 3D-UNet with spatial data augmentation. The 3D UNet takes the image and segments the object.
The model is trained by inputing dense augmented supervised training set. This approach gives a good segmentation with small
training data set.

# How to install

# Instruction
# Training
To train the model, pass the training data directory to train.py The input image should be in [z,y,x,c] format.

# Testing
Measures a dice scores between the prediction and ground truth.
Run test.py


# Citation
3D UNet with spatial data augmentation is used in:
"Few Labeled Atlases are Necessary for Deep-Learning-Based Segmentation." 
Hyeon Woo Lee, Mert R. Sabuncu, and Adrian V. Dalca. 
Neurips ML4HL. arXiv preprint arXiv:1908.04466 (2019).
* Our proposed method in this paper, multiatlas segmentation with semi-supervision can be found from:
voxelmorph.mit.edu

Spatial Transformer Network implementation code is from:
Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018. eprint arXiv:1805.04605


# Contact
For and problems or questions, please send me an email at hl2343@cornell.edu
