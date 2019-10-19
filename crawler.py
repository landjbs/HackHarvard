"""
Responsible for building database of page data from list of URLs.
Outsources URL processing to urlAnalyzer.py
Outsources HTML processing to htmlAnalyzer.py
Outsoucres database definitions to thicctable.py
"""

import re
import cv2
import urllib.request as ur
import numpy as np
from tqdm import tqdm
from queue import Queue
from threading import Thread
from bert_serving.client import BertClient

import utils as u
from processing.cleaner import clean_text, clean_url

bc = BertClient(check_length=True)

def find_bert(BigArray):
    bert = np.zeros((imArray.shape[0], 1024))
    for i in range(imArray.shape[0]):
        bert[i,:256] = imArray[i,:,-2,0]
        bert[i,256:512] = imArray[i,:,-1,0]
        bert[i,512:768] = imArray[i,:,-2,1]
        bert[i,768:] = imArray[i,:,-1,1]
    return bert, imArray[:,:,:-2,:]

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
    """ """
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
                captionEmbedding = bc.encode([cleanCaption])[0]
            except:
                scrapeMetrics.add(True)
                urlQueue.task_done()
                continue

            try:
                url_response = ur.urlopen(cleanUrl, timeout=0.5)
                imArray = np.array(bytearray(url_response.read()),dtype=np.uint8)
                imArray = cv2.imdecode(imArray, cv2.IMREAD_COLOR)
            except Exception as e:
                scrapeMetrics.add(True)
                urlQueue.task_done()
                continue
            if imArray is None:
                scrapeMetrics.add(True)
                urlQueue.task_done()
                continue
            if 256 <= imArray.shape[0] <= 1024:
                if 258 <= imArray.shape[1] <= 1024:
                    # print("cropping")
                    hOffset = int((imArray.shape[0] - 256)/2)
                    wOffset = int((imArray.shape[1] - 258)/2)
                    imArray = imArray[hOffset:hOffset + 256,
                                    wOffset:wOffset + 258,:]
                else:
                    scrapeMetrics.add(True)
                    urlQueue.task_done()
                    continue
            else:
                scrapeMetrics.add(True)
                urlQueue.task_done()
                continue

            imArray[:,-2,0] = captionEmbedding[:256]
            imArray[:,-1,0] = captionEmbedding[256:512]
            imArray[:,-2,1] = captionEmbedding[512:768]
            imArray[:,-1,1] = captionEmbedding[768:]

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

    # iterate over data file
    with open(dataPath, 'r') as dataFile:
        # find number of lines in datafile
        for lineMax, _ in enumerate(dataFile):
            pass
        dataFile.seek(0)

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
                cleanCaption = clean_text(caption)
                cleanUrl = clean_url(lineSplit[0])
                sampleTuple = (cleanCaption, cleanUrl)
                urlQueue.put(sampleTuple)
                lineCounter += 1
                littleIter += 1
            # convert img queue into single numpy array
            imgSize = imgQueue.qsize()
            if imgSize > 0:
                print("IMG")
                imgTensor = np.zeros(shape=(imgSize, 256, 258, 3))
                for i in range(imgSize):
                    print(f'START: {i}')
                    curArray = imgQueue.get()
                    print(curArray)
                    print(imgQueue.qsize())
                    imgTensor[i, :, :, :] = curArray
                    print(f'END: {i}')
                np.save(f'{outFolder}/imgTensor_{lineCounter}', imgTensor)

    print(f'\n{"-"*80}Scraping Complete:\n\tAnalyzed: {scrapeMetrics.count}' \
        f'Errors: {scrapeMetrics.errors}')
    return True


process_caption_data('data/inData/captionsTrain.tsv', 'data/outData/trainArrays', queueDepth=10, workerNum=1)
