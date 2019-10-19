"""
Reads image/caption pairs from coco dataset, encoding captions and images
into sampleTensor.
"""

import json

class CocoObs():
    """ Class to store single observation from coco dataset """
    
    def __init__(self, )

class CocoData():
    """ Class to store observations from coco dataset """

    def __init__(self, cocoPath):
        self.cocoPath = cocoPath
        self.idx = {}


    def read_coco(cocoPath):
        """ Reads captions and images from folders under coco path """
        with open(f'{cocoPath}/annotations/captions_val2014.json', 'r') as annoFile:
            data = json.load(cocoFile)
            for elt in data['annotations']:
                print(elt)
