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
        self.KERNEL_SIZE        =   [5, 5]
        # default convolution stride length
        self.STRIDE             =   [2, 2]
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
        ## TRAINING PARAMS ##
        # number of nodes for dense network at latent stage
        DENSE_NODES     = (8 * 8 * self.EMBEDDING_DIM)
        # shape of reshaped latent dimensional vec
        LATENT_IMAGE_SHAPE = (8, 8, self.EMBEDDING_DIM)
        # momentum of batch norm
        NORM_MOMENTUM   = self.NORM_MOMENTUM
        # rate of dropout
        DROPOUT = self.DROPOUT
        # size of kernel
        KERNEL_SIZE = self.KERNEL_SIZE
        # size of stride
        STRIDE = self.STRIDE

        ## LATENT STAGE ##
        # initialize generator with embedding vector from text
        latent_embedding = Input(shape=(self.EMBEDDING_DIM, ),
                                name='latent_embedding')
        # run dense net over latent vector
        latent_dense = Dense(units=DENSE_NODES,
                            name='latent_dense')(latent_embedding)
        # batch norm latent dense
        latent_batch = BatchNormalization(momentum=NORM_MOMENTUM,
                                          name='batch_latent')(latent_dense)
        # relu activation over latent dense after batch norm
        latent_relu = ReLU(name='relu_latent')(latent_batch)
        # reshape latent dimensions into picture size
        latent_reshape = Reshape(target_shape=LATENT_IMAGE_SHAPE,
                                name='latent_reshape')(latent_relu)
        # run dropout over reshape latent image
        latent_dropout = Dropout(rate=DROPOUT,
                                name='latent_dropout')(latent_reshape)

        ## FIRST UPSAMPLING BLOCK ##
        transpose_1 = Conv2DTranspose(filters=256, kernel_size=KERNEL_SIZE,
                                    padding='same',
                                    strides=STRIDE)(latent_dropout)
        batch_1 = BatchNormalization(momentum=NORM_MOMENTUM,
                                    name='batch_1')(transpose_1)
        relu_1 = ReLU(name='relu_1')(batch_1)

        ## SECOND UPSAMPLING BLOCK ##
        transpose_2 = Conv2DTranspose(filters=64, kernel_size=KERNEL_SIZE,
                                    padding='same', strides=STRIDE)(relu_1)
        batch_2 = BatchNormalization(momentum=NORM_MOMENTUM,
                                    name='batch_2')(transpose_2)
        relu_2 = ReLU(name='relu_2')(batch_2)

        ## THIRD UPSAMPLING BLOCK ##
        transpose_3 = Conv2DTranspose(filters=16, kernel_size=KERNEL_SIZE,
                                    padding='same', strides=STRIDE)(relu_2)
        batch_3 = BatchNormalization(momentum=NORM_MOMENTUM,
                                    name='batch_3')(transpose_3)
        relu_3 = ReLU(name='relu_3')(batch_3)

        ## FOURTH UPSAMPLING BLOCK ##
        transpose_4 = Conv2DTranspose(filters=8, kernel_size=KERNEL_SIZE,
                                    padding='same', strides=STRIDE)(relu_3)
        batch_4 = BatchNormalization(momentum=NORM_MOMENTUM,
                                    name='batch_4')(transpose_4)
        relu_4 = ReLU(name='relu_4')(batch_4)

        ## FIFTH UPSAMPLING BLOCK ##
        transpose_5 = Conv2DTranspose(filters=8, kernel_size=KERNEL_SIZE,
                                    padding='same', strides=STRIDE)(relu_4)
        batch_5 = BatchNormalization(momentum=NORM_MOMENTUM,
                                    name='batch_5')(transpose_5)
        relu_5 = ReLU(name='relu_5')(batch_5)

        ## SIXTH UPSAMPLING BLOCK ##
        transpose_6 = Conv2DTranspose(filters=3, kernel_size=KERNEL_SIZE,
                                    padding='same', strides=STRIDE)(relu_5)
        batch_6 = BatchNormalization(momentum=NORM_MOMENTUM,
                                    name='batch_6')(transpose_6)
        relu_6 = ReLU(name='relu_6')(batch_6)

        model = Model(inputs=latent_embedding, outputs=relu_6)

        # TODO: save model and shit
        print(model.summary())

    def build_descriminator(self):
        """
        Builds desciminator model
        """
        IMG_SHAPE = (512, 512, 3)
        # rate of dropout
        DROPOUT = self.DROPOUT
        # size of kernel
        KERNEL_SIZE = self.KERNEL_SIZE
        # size of stride
        STRIDE = self.STRIDE
        LEAKY_ALPHA = self.LEAKY_ALPHA


        img_in = Input(shape=(IMG_SHAPE, ), name='img_in')
        # first conv block
        conv_1 = Conv2D(filters=16, kernel_size=KERNEL_SIZE,
                        strides=STRIDE, name='conv_1')(img_in)
        relu_1 = LeakyReLU(LEAKY_ALPHA, name=f'relu_1')(conv_1)
        drop_1 = Dropout(rate=DROPOUT, name=f'drop_1')(relu_1)

        # second conv block
        conv_2 = Conv2D(filters=32, kernel_size=KERNEL_SIZE,
                        strides=STRIDE, name='conv_2')(drop_1)
        relu_2 = LeakyReLU(LEAKY_ALPHA, name=f'relu_2')(conv_2)
        drop_2 = Dropout(rate=DROPOUT, name=f'drop_2')(relu_2)

        # third conv block







x = Image_Generator(1,1,1)
x.build_descriminator()
