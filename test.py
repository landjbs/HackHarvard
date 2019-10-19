import processing.image as i

import numpy as np


x = np.load('data/outData/trainArrays/imgTensor_100.npy')


print(x)
print(x.shape)

bertBatch, imBatch = (i.decode_batchArray(x))

print(f'\n\nBert: {bertBatch.shape}]\nIm: {imBatch.shape}')
