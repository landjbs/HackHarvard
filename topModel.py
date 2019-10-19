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
        ## text specs ##
        self.embeddingDim  = 1024
        self.maxTextLen    = maxTextLen
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
