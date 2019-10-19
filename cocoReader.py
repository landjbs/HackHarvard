"""
Reads image/caption pairs from coco dataset, encoding captions and images
into sampleTensor.
"""

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

# import json
#
# with open('data/inData/annotations/stuff_train2017.json', 'r') as annoFile:
#     data = json.load(annoFile)['annotations']
#     for e in data:
#         print(e['bbox'])
#

# pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
# pip3 install Cython
