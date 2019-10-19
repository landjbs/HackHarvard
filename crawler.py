"""
Responsible for building database of page data from list of URLs.
Outsources URL processing to urlAnalyzer.py
Outsources HTML processing to htmlAnalyzer.py
Outsoucres database definitions to thicctable.py
"""

from queue import Queue
from threading import Thread
from time import time
import re
import requests
import shutil
import numpy as np
import cv2
import urllib
import utils as u
from bert_serving.client import BertClient
from processing.cleaner import clean_text, clean_url

bc = BertClient(check_length=True)

def find_bert(imArray):
    bert = np.zeros(1024)
    bert[:256] = imArray[:,-2,0]
    bert[256:512] = imArray[:,-1,0]
    bert[512:768] = imArray[:,-2,1]
    bert[768:] = imArray[:,-1,1]
    return bert

class Metrics():
    """ Class to keep track of scrape progress """
    def __init__(self):
        self.count = 0
        self.errors = 0

    def add(self, error=False):
        self.count += 1
        if error:
            self.errors += 1



def process_caption_data(dataPath, outFolder, queueDepth=10000, workerNum=30):
    """ """
    u.safe_make_folder(outFolder)

    scrapeMetrics = Metrics()
    urlQueue = Queue(maxsize=queueDepth)
    imgQueue = Queue(maxsize=(queueDepth+1))
    lineCounter = 0

    def worker():
        while True:
            # pop top url from queue
            cleanCaption, cleanUrl = imgQueue.get()
            captionEmbedding = np.array(bc.encode([cleanCaption])[0])

            try:
                # print(cleanedURL)
                url_response = urllib.request.urlopen(cleanUrl,timeout=0.5)
                # print("accessed website")
                imArray = np.array(bytearray(url_response.read()),dtype=np.uint8)
                imArray = cv2.imdecode(imArray, cv2.IMREAD_COLOR)
                # print("built arrays")
            except:
                scrapeMetrics.add(True)
            if imArray is None:
                scrapeMetrics.add(True)
            if 256 <= imArray.shape[0] <= 1024:
                if 258 <= imArray.shape[1] <= 1024:
                    # print("cropping")
                    hOffset = int((imArray.shape[0] - 256)/2)
                    wOffset = int((imArray.shape[1] - 258)/2)
                    imArray = imArray[hOffset:hOffset + 256, wOffset:wOffset + 258,:]
                else:
                    scrapeMetrics.add(True)
            else:
                scrapeMetrics.add(True)

            imArray[:,-2,0] = captionEmbedding[:256]
            imArray[:,-1,0] = captionEmbedding[256:512]
            imArray[:,-2,1] = captionEmbedding[512:768]
            imArray[:,-1,1] = captionEmbedding[768:]

            imgQueue.put(imArray)
            urlQueue.task_done()
            imgQueue.task_done()
            print(f'URLs Analyzed: {scrapeMetrics.count} | Errors: {errors}',
                    end='\r')

    # spawn workerNum workers
    for _ in range(workerNum):
        t = Thread(target=worker)
        t.daemon = True
        t.start()

    # iterate over data file
    with open(dataPath, 'r') as dataFile:
        # find number of lines in datafile
        for lineMax, _ in enumerate(dataFile):
            pass
        dataFile.seek(0)

        while lineCounter < lineMax:
            for i, line in enumerate(dataFile):
                if ((i % queueDepth) == 0 and (i != 0)):
                    break
                lineSplit = re.split('\t', line)
                assert (len(lineSplit)==2), ('line expected length 2, but found '
                                            f'length {len(lineSplit)}')
                caption = lineSplit.pop(0)
                cleanCaption = clean_text(caption)
                cleanUrl = clean_url(lineSplit[0])
                sampleTuple = (cleanCaption, cleanUrl)
                urlQueue.put(sampleTuple)
            # convert img queue into single numpy array
            imgSize = imgQueue.qsize()
            imgTensor = np.zeros(shape=(imgSize, 256, 258, 3))
            for i in range(imgSize):
                curArray = imgQueue.get()
                imgTensor[i, :, :, :] = curArray
            np.save(f'{outFolder}/imgTensor_{i}', imgTensor)

    print(f'\n{"-"*80}Scraping Complete:\n\tAnalyzed: {scrapeMetrics.count}' \
        f'Errors: {scrapeMetrics.errors}')
    return True
