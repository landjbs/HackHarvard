"""
Reads caption data from google conceptual captions dataset. Uses multithreading
and url access to fetch images and encode as tensors saved under outPath.
"""

import re
import cv2
import numpy as np
from tqdm import tqdm
from queue import Queue
import urllib.request as ur
from threading import Thread
from bert_serving.client import BertClient

import utils as u
import processing.text as textProcessing
import processing.image as imageProcessing

bc = BertClient(check_length=True)

class Metrics():
    """ Class to keep track of scrape progress """
    def __init__(self):
        self.count = 0
        self.errors = 0

    def add(self, error=False):
        self.count += 1
        if error:
            self.errors += 1

def process_caption_data(dataPath, outFolder, queueDepth=100, workerNum=3):
    """
    Code to process conceptual captions data by encoding caption text and
    fetching images from url. Saves numpy batch files to outFolder.
    """
    u.safe_make_folder(outFolder)

    scrapeMetrics = Metrics()
    urlQueue = Queue(maxsize=queueDepth+1)
    imgQueue = Queue(maxsize=(queueDepth+1))
    lineCounter = 0

    def worker():
        while True:
            # pop top url from queue
            cleanCaption, cleanUrl = urlQueue.get()
            try:
                # fetch cls token of caption from bert server
                capVec = bc.encode([cleanCaption])[0]
                # fetch raw image array from url
                rawIm = np.array(bytearray(ur.urlopen(cleanUrl, timeout=0.5)),
                                dtype=np.uint8)
                # decode im array
                rawIm = cv2.imdecode(imArray, cv2.IMREAD_COLOR)
                # filter and resize image
                imArray = imageProcessing.filter_image(rawIm)
                # encode caption vector and image array into single tensor
                sampleTensor = imageProcessing.embed_bert(capVec, imArray)
            except:
                scrapeMetrics.add(True)
                urlQueue.task_done()
                continue
            finally:
                # log and close task
                scrapeMetrics.add(False)
                print(f'URLs Analyzed: {scrapeMetrics.count} |'
                    f'Errors: {scrapeMetrics.errors}', end='\r')
                imgQueue.put(imArray)
                urlQueue.task_done()
                imgQueue.task_done()

    # spawn workerNum workers
    for _ in range(workerNum):
        t = Thread(target=worker)
        t.daemon = True
        t.start()

    # iterate over data file to count line num
    with open(dataPath, 'r') as dataFile:
        # find number of lines in datafile
        for lineMax, _ in enumerate(dataFile):
            pass
        dataFile.seek(0)
        # iterate over entire file
        while lineCounter < lineMax:
            littleIter = 0
            for line in dataFile:
                if littleIter >= queueDepth:
                    littleIter = 0
                    urlQueue.join()
                    break
                lineSplit = re.split('\t', line)
                assert (len(lineSplit)==2), ('line expected length 2, but found '
                                            f'length {len(lineSplit)}')
                caption = lineSplit.pop(0)
                cleanCaption = textProcessing.clean_text(caption)
                cleanUrl = textProcessing.clean_url(lineSplit[0])
                sampleTuple = (cleanCaption, cleanUrl)
                urlQueue.put(sampleTuple)
                print(f'\t\t{lineCounter}', end='\r')
                lineCounter += 1
                littleIter += 1
            # convert img queue into single numpy array
            imgSize = imgQueue.qsize()
            if imgSize > 0:
                imgTensor = np.zeros(shape=(imgSize, 256, 258, 3))
                for i in range(imgSize):
                    curArray = imgQueue.get()
                    imgTensor[i, :, :, :] = curArray
                np.save(f'{outFolder}/imgTensor_{lineCounter}', imgTensor)

    print(f'\n{"-"*80}Scraping Complete:\n\tAnalyzed: {scrapeMetrics.count}' \
        f'Errors: {scrapeMetrics.errors}')
    return True


process_caption_data('data/inData/captionsTrain.tsv', 'data/outData/trainArrays', queueDepth=100, workerNum=1)
