#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Hyeonwoo Lee",
    author_email='hyeonwoo610@gmail.com',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="3D image segmentation with UNet and spatial data augmentation",
    entry_points={
        'console_scripts': [
            'segmentation_3D=segmentation_3D.cli:main',
        ],
    },
    install_requires=[
        'keras >= 2.31',
        'numpy >= 1.18.1',
        'pillow >= 7.0.0',
        'scikit-image >= 0.16.2',
        'scipy >= 1.4.1',
        'tensorflow >= 2.1.0',
        'tqdm >= 4.42.1',
        'keras-tqdm >= 2.0.1'
    ],
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='segmentation_3D',
    name='segmentation_3D',
    packages=find_packages(include=['segmentation_3D', 'segmentation_3D.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/lhw610/segmentation_3D',
    version='0.1.0',
    zip_safe=False,
)
