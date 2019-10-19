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

def process_caption_data(dataPath, outFolder, queueDepth=10000, workerNum=30):
    """ """
    u.safe_make_folder(outFolder)

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
                return None
            if imArray is None:
                return None
            if 256 <= imArray.shape[0] <= 1024:
                if 258 <= imArray.shape[1] <= 1024:
                    # print("cropping")
                    hOffset = int((imArray.shape[0] - 256)/2)
                    wOffset = int((imArray.shape[1] - 258)/2)
                    imArray = imArray[hOffset:hOffset + 256, wOffset:wOffset + 258,:]
                else:
                    return None
            else:
                return None
            # imArray[:,-2,0] = captionEmbedding[:256]
            # imArray[:,-1,0] = captionEmbedding[256:512]
            # imArray[:,-2,1] = captionEmbedding[512:768]
            # imArray[:,-1,1] = captionEmbedding[768:]
            return imArray

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
            for i, line in enumerate(dataFile):
                lineSplit = re.split('\t', line)
                assert (len(lineSplit)==2), ('line expected length 2, but found '
                                            f'length {len(lineSplit)}')
                caption = lineSplit.pop(0)
                cleanCaption = clean_text(caption)
                cleanUrl = clean_url(lineSplit[0])
                sampleTuple = (cleanCaption, cleanUrl)
                urlQueue.put(sampleTuple)








def url_to_nice_array(cleanedURL):
    try:
        # print(cleanedURL)
        url_response = urllib.request.urlopen(cleanedURL,timeout=0.5)
        # print("accessed website")
        imArray = np.array(bytearray(url_response.read()),dtype=np.uint8)
        imArray = cv2.imdecode(imArray, cv2.IMREAD_COLOR)
        # print("built arrays")
    except:
        return None
    if imArray is None:
        return None
    if 256 <= imArray.shape[0] <= 1024:
        if 258 <= imArray.shape[1] <= 1024:
            # print("cropping")
            hOffset = int((imArray.shape[0] - 256)/2)
            wOffset = int((imArray.shape[1] - 258)/2)
            imArray = imArray[hOffset:hOffset + 256, wOffset:wOffset + 258,:]
        else:
            return None
    else:
        return None
    return imArray


def scrape_urlList(urlList, runTime=100000000, queueDepth=1000000, workerNum=2):
    """
    Rescursively crawls internet from starting urlList and ends after runTime
    seconds.
    """

    # queue to hold urlList
    imgQueue = Queue(queueDepth)

    # find time at which to stop analyzing
    stopTime = round(time() + runTime)

    def enqueue_urlList(captionFile):
        """
        Cleans and enqueues URLs contained in urlList, checking if
        previously scraped
        """
        with open(filePath, 'r') as captionFile:
            for line in captionFile:
                lineSplit = re.split('\t', line)
                assert (len(lineSplit)==2), ('line expected length 2, but found '
                                            f'length {len(lineSplit)}')
                # get caption
                lineCap = lineSplit.pop(0)
                # get url
                cleanedURL = re.sub(secureMatcher, "http", lineSplit[0].strip("\n"))
                imgQueue.put((lineCap,cleanedURL))

    def worker():
        """ Scrapes popped URL from urlQueue and stores data in database"""
        while True:
            # pop top url from queue
            cleanedURL = imgQueue.get()

            try:
                imArray = url_to_array(cleanedURL)
            except:
                imgQueue.task_done()
                continue

            # log progress
            print(f"\tURLs ANALYZED: {scrapeMetrics.count} | Errors: {scrapeMetrics.errors} | Queue Size: {urlQueue.qsize()}", end="\r")
            # signal completion
            imgQueue.task_done()

    # spawn workerNum workers
    for _ in range(workerNum):
        t = Thread(target=worker)
        t.daemon = True
        t.start()

    # load cleaned initial urls into url_queue
    urlList = list(map(lambda url:fix_url(url, url), urlList))
    enqueue_urlList(urlList)

    # ensure all urlQueue processes are complete before proceeding
    urlQueue.join()

    return True
