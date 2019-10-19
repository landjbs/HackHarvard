"""
Reads image/caption pairs from coco dataset, encoding captions and images
into sampleTensor.
"""

import re
import json
import numpy as np
from tqdm import tqdm
from cv2 import imread
import matplotlib.pyplot as plt

import utils as u

# re matcher to pull id from image path
pathIdMatcher = re.compile(r'(?<=0)[1-9]\d*(?=.jpg)')

class CocoData():
    """ Class to store observations from coco dataset """
    def __init__(self, cocoPath):
        assert isinstance(cocoPath, str) or isinstance(cocoPath, None), ('coco '
            f'path expected either string or None, but found {type(cocoPath)}.')

        self.cocoPath = cocoPath

        def coco_reader(cocoPath):
            """ Reads coco path folders into object index """
            # config string
            captionPath = f'{cocoPath}/annotations/captions_train2014.json'
            imageFolder = f'{cocoPath}/train2014'
            # build dict mapping image_id to string of image
            path_to_id = lambda path : int(re.findall(pathIdMatcher, path)[0])
            imIdx = {path_to_id(path) : path
                    for path in u.os.listdir(f'{cocoPath}/train2014')}

            with open(captionPath, 'r') as captionFile:
                captionData = json.load(captionFile)
                capNum = len(captionData['annotations'])
                for i, example in tqdm(enumerate(captionData['annotations']), total=capNum):
                    # if i > 100:
                    #     break
                    imgId = example['image_id']
                    imArray = imread(f"{imageFolder}/{imIdx[imgId]}")[:,:,::-1]
                    print(imArray.shape, end='\n\n')
                    yield (example['id'], example['caption'], imArray)

        if cocoPath:
            self.trainIdx = {i : dataTup for i, dataTup
                            in tqdm(enumerate(coco_reader(cocoPath)))}
        else:
            self.trainIdx = None

    def save(self, path):
        """ Saves trainIdx to path """
        u.safe_make_folder(path)
        u.save(self.trainIdx, f'{path}/trainIdx.sav', display=False)

x = CocoData('data/inData/coco2014')
# print(x.trainIdx)
# x.save('test')

# x = CocoData()
