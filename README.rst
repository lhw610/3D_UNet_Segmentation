===============
3D_Segmentation
===============


.. image:: https://img.shields.io/pypi/v/segmentation_3D.svg
        :target: https://pypi.python.org/pypi/segmentation_3D

.. image:: https://img.shields.io/travis/lhw610/segmentation_3D.svg
        :target: https://travis-ci.com/lhw610/segmentation_3D

.. image:: https://readthedocs.org/projects/segmentation-3D/badge/?version=latest
        :target: https://segmentation-3D.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




3D image segmentation with UNet and spatial data augmentation


* Free software: GNU General Public License v3
* Documentation: https://segmentation-3D.readthedocs.io.

# 3D Segmentation
This is an implementation of 3D-UNet with spatial data augmentation. The 3D UNet takes the image and segments the object.
It can perform binary segmenation, or even multi label segmentation. The full 3D deformation makes this model robust in low supervised setting.

Install instruction
-------
To install this package,
1. git clone the package
2. cd ./3D_UNet_segmentaion
2. pip install -e .

Train the model
-------
To train the model, pass the training data directory to ./segmentation_3D/train.py 
The input image should be in [z,y,x,channel] format.
run train.py by passing required parameters like following Example

Example:
python train.py --source /path/to/source/directory 
                --target /path/to/source/directory 
                --data_type tiff
                --ch 2
                --save_name demo 
                --iters 100 
                --save_iters 10 
                --batch 20



Inference
-------
Measures a dice scores between the prediction and ground truth.
Run test.py


Citation
-------
3D UNet with spatial data augmentation is used in:
"Few Labeled Atlases are Necessary for Deep-Learning-Based Segmentation." 
Hyeon Woo Lee, Mert R. Sabuncu, and Adrian V. Dalca. 
Neurips ML4HL. arXiv preprint arXiv:1908.04466 (2019).
* Our proposed method in this paper, multi-atlas segmentation with semi-supervision can be found from:
voxelmorph.mit.edu

Spatial Transformer Network implementation code is from:
Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018. eprint arXiv:1805.04605

Contacts
-------
For and problems or questions, please send me an email at hl2343@cornell.edu

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
