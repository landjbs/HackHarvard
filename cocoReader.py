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
import processing.text as text
import processing.image as image

# re matcher to pull id from image path
pathIdMatcher = re.compile(r'(?<=0)[1-9]\d*(?=.jpg)')

class CocoData():
    """ Class to store observations from coco dataset """
    def __init__(self, cocoPath=None):
        assert (isinstance(cocoPath, str) or (cocoPath==None)), ('cocoPath '
            f'expected either string or None, but found {type(cocoPath)}.')

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
                i = 0
                for example in tqdm(captionData['annotations'], total=capNum):
                    i += 1

                    if i > 10:
                        break

                    imgId = example['image_id']
                    captionText = text.clean_text(example['caption'])
                    imArray = imread(f"{imageFolder}/{imIdx[imgId]}")[:,:,::-1]
                    # encode caption and clean image
                    try:
                        captionVec = text.text_to_cls(captionText)
                        cleanedIm = image.filter_image(imArray)
                        yield (captionVec, captionVec, imArray)
                    except:
                        yield None
                        
        if cocoPath:
            error_filter = lambda elt : elt != None
            trainIdx = {i : dataTup for i, dataTup
                        in tqdm(enumerate(coco_reader(cocoPath)))
                        if not error_filter(dataTup)}
            self.trainIdx = trainIdx
        else:
            self.trainIdx = None

    def __str__(self):
        if self.trainIdx != None:
            return f'<CocoData Obj | TRAIN_NUM={len(self.trainIdx)}>'
        else:
            return f'<CocoData Obj | UNINITIALIZED>'

    def save(self, path):
        """ Saves to path """
        u.safe_make_folder(path)
        u.save(self.trainIdx, f'{path}/trainIdx.sav', display=False)

    def load(self, path):
        """ Loads from path """
        u.assert_type('path', path, str)
        assert u.os.path.exists(path), f'path "{path}" not found.'
        self.trainIdx = u.load(f'{path}/trainIdx.sav')

x = CocoData('data/inData/coco2014')
print(x.trainIdx)
x.save('test')
