"""
Reads image/caption pairs from coco dataset, encoding captions and images
into sampleTensor.
"""

import re
import json
import numpy as np
import matplotlib.pyplot as plt

import utils as u

# re matcher to pull id from image path
pathIdMatcher = re.compile(r'(?<=0)[1-9]\d*(?=.jpg)')

class CocoData():
    """ Class to store observations from coco dataset """
    def __init__(self, cocoPath):
        self.cocoPath = cocoPath

        def coco_reader(cocoPath):
            """ Reads coco path folders into object index """
            # config string
            captionPath = f'{cocoPath}/annotations/captions_train2014.json'
            # build dict mapping image_id to string of image
            path_to_id = lambda path : int(re.findall(pathIdMatcher, path)[0])
            imageIdx = {path_to_id(path) : path
                        for path in u.os.listdir(f'{cocoPath}/train2014')}

            with open(captionPath, 'r') as captionFile:
                for example in captionFile:
                    caption = example['caption']
                    exampleId = example['id']
                    imgId = example['image_id']
                    print(imageIdx[imgId])

        self.trainIdx = coco_reader(cocoPath)


x = CocoData('data/inData/coco2014')
