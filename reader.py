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


def sample_iter

def read_dataset(fileName):
    """
    Reads conceptual caption dataset from fileName into list of img/caption
    pairs
    """
    with open(fileName, 'r') as captionFile:
        for line in captionFile:
            # split line by tab
            lineSplit = re.split('\t', line)
            assert (len(lineSplit)==2), ('line expected length 2, but found '
                                         f'length {len(lineSplit)}')
            # pull
            lineCap = lineSplit.pop(0)
            print(lineCap)
            imgFetch = requests.get(lineSplit[0])
            img = Image.open(BytesIO(imgFetch.content))
            print(img)


read_dataset('data/inData/captionsTrain.tsv')
