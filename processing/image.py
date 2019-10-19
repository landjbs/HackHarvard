"""
Utils for cleaning and processing all images encountered by models.
"""


def filter_image(imArray, outDim=256, upperBound=1024):
    """
    Filters imArray to outDim, filtering those with dimensions
    outside of outDim and upperBound. Raises error if
    """
    imShape = imArray.shape
    if ((outDim <= imShape[0] <= upperBound)
        and ((outDim + 2) <= imShape[1] <= upperBound)):
            hOffset = int((imShape[0] - outDim)/2)
            wOffset = int((imShape[1] - outDim)/2)
            imArray = imArray[hOffset:hOffset + outDim,
                            wOffset:wOffset + outDim, :]
    else:
        raise InputError(f'Image has invalid dims {imShape}.')

def embed_bert(bertVec, imArray):
    """ Embeds bert caption vec in image array and returns single tenosr """
