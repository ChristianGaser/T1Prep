#!/usr/bin/env python

import sys
import setuptools

python_version = sys.version[:3]

if (python_version != '3.8') & (python_version != '3.9'):
    raise Exception('Setup.py only works with python version 3.8 or 3.9, not {}'.format(python_version))

else:

    with open('requirements_python' + python_version + '.txt') as f:
        required_packages = [line.strip() for line in f.readlines()]

    print(setuptools.find_packages())

    setuptools.setup(name='T1Prep',
                     version='0.1.0',
                     license='Apache 2.0',
                     description='T1 PREProcessing Pipeline (aka PyCAT)',
                     author='Christian Gaser <christian.gaser@uni-jena.de>',
                     url='https://github.com/ChristianGaser/T1Prep',
                     keywords=['segmentation', 'domain-agnostic', 'brain'],
                     packages=setuptools.find_packages(),
                     python_requires='>=3.8',
                     install_requires=required_packages,
                     include_package_data=True)