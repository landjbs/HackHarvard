from keras.models import Model, Sequential
from keras.layers import (Input, Conv2D, Activation, LeakyReLU, Dropout,
                            Flatten, Dense, BatchNormalization, ReLU,
                            UpSampling2D, Conv2DTranspose, Reshape)

import utils as u


class Image_Generator():
    """
    Top-level model comprising summarizer, generator, discriminator, and
    describer.
    """

    def __init__(self, rowNum, colNum, maxTextLen):
        # assertions
        u.assert_type('rowNum', rowNum, int)
        u.assert_type('colNum', colNum, int)
        u.assert_type('maxTextLen', maxTextLen, int)
        u.assert_pos('rowNum', rowNum)
        u.assert_pos('colNum', colNum)
        u.assert_pos('maxTextLen', maxTextLen)
        ## text specs ##
        self.maxTextLen    = maxTextLen
        self.EMBEDDING_DIM  = 1024
        ## image specs ##
        self.rowNum         = rowNum
        self.colNum         = colNum
        self.CHANNEL_NUM    = 3
        ## object info ##
        self.curIter        = 0
        self.summarizer     = None
        self.generator      = None
        self.discriminator  = None
        self.describer      = None
        self.init           = False
        ## training specs ##
        # default first-layer filter depth of discriminator
        DIS_DEPTH               =   64
        self.DIS_DEPTH          =   DIS_DEPTH
        self.GEN_DEPTH          =   DIS_DEPTH * 4
        # default dropout; should prevent memorization
        self.DROPOUT            =   0.2
        # default kernel size
        self.KERNEL_SIZE        =   5
        # default convolution stride length
        self.STRIDE             =   2
        # default alpha of LeakyReLU activation in discriminator
        self.LEAKY_ALPHA        =   0.2
        # dimensions of generator latent space
        self.LATENT_DIMS        =   100
        # default momentum for adjusting mean and var in generator batch norm
        self.NORM_MOMENTUM      =   0.9

    def __str__(self):
        return (f'<Image_Generator Model | INIT={self.init} '
                f'| ITER={self.curIter}>')

    def build_summarizer(self):
        """
        Builds and compiles summarizer model to read BERT matrix into vector of
        EMBEDDING_DIM dimensions.
        """
        # TODO: IMP SUMMARIZER
        pass

    def build_generator(self):
        """
        Builds generator to convert latent initialization vector of
        EMBEDDING_DIM dimensions to third-rank tensor of
        shape (rowNum, colNum, channelNum)
        """
        # initialize generator with embedding vector from text
        latent_embedding = Input(shape=self.EMBEDDING_DIM,
                                name='latent_embedding')
        # 
