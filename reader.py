"""
Code for reading from conceptual captions dataset including img fetching.
Builds dataset of image caption pairs as tuples of (captionStr, imageTensor).
"""

# from processing import clean_text

import re
import requests
from io import BytesIO
from PIL import Image

# response = requests.get(url)
# img = Image.open(BytesIO(response.content))


def sample_iter(n):
    for i in range(n):
        yield i


def read_dataset(filePath, batchSize):
    """
    Reads conceptual caption dataset from fileName into list of img/caption
    pairs
    """
    secureMatcher = re.compile(r"https")
    with open(filePath, 'r') as captionFile:
        count = 0
        for line in captionFile:
            count += 1
            if count == 10:
                break
            # split line by tab
            lineSplit = re.split('\t', line)
            assert (len(lineSplit)==2), ('line expected length 2, but found '
                                        f'length {len(lineSplit)}')
            lineCap = lineSplit.pop(0)
            cleanedURL = re.sub(secureMatcher, "http", lineSplit[0])
            imgFetch = requests.get(cleanedURL, auth=('user', 'pass'))
            print(cleanedURL)
            print(imgFetch)
            try:
                img = Image.open(BytesIO(imgFetch.content))
            except IOError:
                continue
            print(img)


read_dataset('data/inData/captionsTrain.tsv', 1)
