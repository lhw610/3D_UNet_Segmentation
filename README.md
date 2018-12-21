# Patch-based_3D_U-Net
Divide a 3D volume into several patches. It trains the individual patches then predicts the label. 
The labels for the overlapping region of patches are determined by calculating the average probability map.

# Instruction
# Training
To train the model, set the training data directory in training.py and run the file. The weights of the model will be save to the models/ folder

# Testing
Measures a dice scores between the prediction and ground truth.
Run test.py


# neuron, patchlib, and medipy toolbox used in code from..
Dr.Adrian Dalca's voxelmoprh
https://github.com/voxelmorph/voxelmorph

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018. eprint arXiv:1805.04605

An Unsupervised Learning Model for Deformable Medical Image Registration
Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
CVPR 2018. eprint arXiv:1802.02604

# Contact
For and problems or questions, please send me an email at hl2343@cornell.edu
