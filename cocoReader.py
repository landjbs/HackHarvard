"""
Reads image/caption pairs from coco dataset, encoding captions and images
into sampleTensor.
"""

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
