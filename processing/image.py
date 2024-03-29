"""
Utils for cleaning and processing all images encountered by models.
"""

import numpy as np
from wand.image import Image

# LANDON BERT
# bert-serving-start -model_dir /Users/landon/Desktop/bertLarge -num_worker=3

def load_and_filter_image(imPath):
    with Image(filename=imPath) as img:
        rowLen, colLen = img.size
        if 0.6 < (rowLen / colLen) < 1.51:
            img.resize(512, 512)
            return np.array(img)
        else:
            raise ValueError(f'Image has invalid dims ({rowLen}, {colLen}).')


def filter_image(imArray, outDim, upperBound=1024):
    """
    Filters imArray to outDim, filtering those with dimensions
    outside of outDim and upperBound. Raises error if
    """
    imShape = imArray.shape
    if ((outDim <= imShape[0] <= upperBound)
        and (outDim <= imShape[1] <= upperBound)):
        hOffset = int((imShape[0] - outDim)/2)
        wOffset = int((imShape[1] - outDim)/2)
        imArray = imArray[hOffset:hOffset + outDim, wOffset:wOffset + outDim, :]
        return (imArray / 255)
    else:
        raise ValueError(f'Image has invalid dims {imShape}.')


def embed_bert(captionEmbedding, imArray):
    """ Embeds bert caption vec in imArray and returns single tenosr """
    imArray[:,-2,0] = captionEmbedding[:256]
    imArray[:,-1,0] = captionEmbedding[256:512]
    imArray[:,-2,1] = captionEmbedding[512:768]
    imArray[:,-1,1] = captionEmbedding[768:]
    return imArray


def decode_batchArray(batchArray):
    """ Decodes tensor of caption vector and image array across batch """
    bert = np.zeros((batchArray.shape[0], 1024))
    for i in range(batchArray.shape[0]):
        bert[i,:256] = batchArray[i,:,-2,0]
        bert[i,256:512] = batchArray[i,:,-1,0]
        bert[i,512:768] = batchArray[i,:,-2,1]
        bert[i,768:] = batchArray[i,:,-1,1]
    return bert, batchArray[:,:,:-2,:]
