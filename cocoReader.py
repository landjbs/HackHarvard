"""
Reads image/caption pairs from coco dataset, encoding captions and images
into sampleTensor.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

import utils as u

class CocoData():
    """ Class to store observations from coco dataset """
    def __init__(self, cocoPath):
        self.cocoPath = cocoPath

        def coco_reader(cocoPath):
            """ Reads coco path folders into object index """
            # config string to access captions and index to access images
            captionPath = f'{cocoPath}/annotations/captions_train2014.json'
            imagePathIter = u.os.listdir(f'{cocoPath}/train2014')

            for imagePath in imagePathIter:
                print(f'{cocoPath}/train2014/{imagePath}')

            # with open(captionPath, 'r') as captionFile:
            #     for sample in captionFile:
            #         print(sample)

        self.trainIdx = coco_reader(cocoPath)


x = CocoData('data/inData/coco2014')
