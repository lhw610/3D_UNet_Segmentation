3D UNet Segmentation
===============
3D image segmentation with UNet and spatial data augmentation


* Free software: GNU General Public License v3

Intro
-------
This is an implementation of 3D-UNet with spatial data augmentation. The 3D UNet takes the 3D image and segments the object. <br />
It can perform binary segmenation, or even multi label segmentation. The full 3D deformation makes this model robust in low supervised setting.

Install instruction
-------
To install this package,

1. git clone https://github.com/lhw610/3D_UNet_segmentaion.git

2. cd ./3D_UNet_segmentaion

3. pip install -e .

pip install -e . will install required the dependencies. I recommend to create new conda environment before install this package.

Train the model
-------
To train the model, pass the training data directory to ./segmentation_3D/train.py <br />
The input image should be in [z,y,x,channel] format. <br />
run train.py by passing required arguments like following Example <br />

Example:
python train.py --source /path/to/source/directory --target /path/to/target/directory --data_type tiff --ch 2 --save_name demo --iters 100 --save_iters 10 --batch 20

Inference
-------
To inference, pass the test image directory and other arugmentsto ./segmentation_3D/test.py <br />
The input image should be in [z,y,x,channel] format. <br />
The segmented image will be saved to ./inference/save_dir 

Example:
python test.py --dir /path/to/test/image/directory --data_type tiff --ch 2 --model demo --iters 100 --save_dir directory name to save result


Citation
-------
This 3D UNet with spatial data augmentation used in segmentation model comparison in low supervised setting:

"Few Labeled Atlases are Necessary for Deep-Learning-Based Segmentation." <br />
Hyeon Woo Lee, Mert R. Sabuncu, and Adrian V. Dalca. <br />
arXiv preprint arXiv:1908.04466 (2019).

* Our proposed method in this paper, multi-atlas segmentation with semi-supervision can be found from:
voxelmorph.mit.edu

Spatial Transformer Network implementation code is from:

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration <br />
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu <br />
MICCAI 2018. eprint arXiv:1805.04605

Contacts
-------
For and problems or questions, please send me an email at hl2343@cornell.edu

Credits
-------

This package was created with Cookiecutter and the `audreyr/cookiecutter-pypackage` project template.