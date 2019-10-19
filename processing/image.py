"""
Utils for cleaning and processing all images encountered by models.
"""


def filter_image(imArray, outDim=256, ):
    """
    Filters imArray to outDim, filtering those with dimensions
    outside of upper- and higherBound.
    """
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
