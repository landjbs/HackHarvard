"""
The Image_Generator is a new generation of Deep Convolutional GAN, which
uses Recurrent Neural Networks and Attention Mechanisms to summarize text
as a latent dimensional initialization vector for a generative network.
Through a multi-layered and morphing training-process, the summarizer-generator
pair is taught to capture the important details of an input text and to use
them to construct and image that is both realistic and topical.
"""

from keras.models import Model, Sequential
from keras.layers import (Input, Conv2D, Activation, LeakyReLU, Dropout,
                            Flatten, Dense, BatchNormalization, ReLU,
                            UpSampling2D, Conv2DTranspose, Reshape)
import keras.backend as K
from keras.optimizers import RMSprop

import utils as u


class Image_Generator():
    """
    Top-level model comprising summarizer, generator, discriminator, and
    describer.
    """

    def __init__(self, maxTextLen):
        # assertions
        u.assert_type('maxTextLen', maxTextLen, int)
        u.assert_pos('maxTextLen', maxTextLen)
        ## text specs ##
        self.maxTextLen    = maxTextLen
        self.EMBEDDING_DIM  = 1024
        ## image specs ##
        self.IMG_SHAPE = (512, 512, 3)
        ## object info ##
        self.curIter                = 0
        # model architectures
        self.summarizerStruct       = None
        self.generatorStruct        = None
        self.discriminatorStruct    = None
        self.describerStruct        = None
        # model compilations
        self.discriminatorModel     = None
        self.describerModel         = None
        self.adversarialModel       = None
        self.creativeModel          = None
        self.initizalized           = False
        ## training specs ##
        # default dropout to prevent memorization
        self.DROPOUT            =   0.2
        # default kernel size
        self.KERNEL_SIZE        =   [5, 5]
        # default convolution stride length
        self.STRIDE             =   [2, 2]
        # default alpha of LeakyReLU activation in discriminator
        self.LEAKY_ALPHA        =   0.2
        # default momentum for adjusting mean and var in generator batch norm
        self.NORM_MOMENTUM      =   0.9

    def __str__(self):
        return (f'<Image_Generator Model | INIT={self.init} '
                f'| ITER={self.curIter}>')

    ## CUSTOM LOSS FUNCTIONS, OPTIMIZERS, AND LR SCALERS ##
    def distance_loss(layer):
        """ Custom loss for euclidean distance minimization """
        def loss(y_true,y_pred):
            return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
        return loss

    ## MODEL BUILDING ##
    def build_summarizer(self):
        """
        Builds and compiles summarizer model to read BERT matrix into vector of
        EMBEDDING_DIM dimensions.
        """
        # matix of word encodings
        word_inputs = Input(shape=(self.BERT_DIM, self.MAX_CAPTION_LEN),
                            name='word_encodings')
        # vector of cls
        cls_input = Input(shape=(self.BERT_DIM, ), name='cls_input')
        # define structure of bidirectional lstm
        gru = Bidirectional(GRU(units=self.LATENT_DIM, activation='relu',
                                    return_state=True))
        # get outputs of lstm run over word_inputs with cls initial state
        lstm_out, hidden, cell = gru(word_inputs, initial_state=cls_input)
        summarizerStructure = Model(inputs=[cls_input, word_inputs],
                                    outputs=cell)

    def build_generator(self):
        """
        Builds generator to convert latent initialization vector of
        EMBEDDING_DIM dimensions to third-rank tensor of
        shape (rowNum, colNum, channelNum)
        """
        ## TRAINING PARAMS ##
        DENSE_NODES     = (8 * 8 * self.EMBEDDING_DIM)
        LATENT_IMG_SHAPE = (8, 8, self.EMBEDDING_DIM)
        NORM_MOMENTUM   = self.NORM_MOMENTUM
        DROPOUT = self.DROPOUT
        KERNEL_SIZE = self.KERNEL_SIZE
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
        latent_reshape = Reshape(target_shape=LATENT_IMG_SHAPE,
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
        # save model
        model = Model(inputs=latent_embedding, outputs=relu_6)
        self.generatorStruct = model
        return True

    def build_discriminator(self):
        """
        Builds structure for the discriminator model, which uses convolutional
        and dense networks to output sigmoid-normed scalar indicating belief
        that an image is real.
        """
        IMG_SHAPE = self.IMG_SHAPE
        DROPOUT = self.DROPOUT
        KERNEL_SIZE = self.KERNEL_SIZE
        STRIDE = self.STRIDE
        LEAKY_ALPHA = self.LEAKY_ALPHA
        # first conv block
        img_in = Input(shape=IMG_SHAPE, name='img_in')
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
        conv_3 = Conv2D(filters=64, kernel_size=KERNEL_SIZE,
                        strides=STRIDE, name='conv_3')(drop_2)
        relu_3 = LeakyReLU(LEAKY_ALPHA, name=f'relu_3')(conv_3)
        drop_3 = Dropout(rate=DROPOUT, name=f'drop_3')(relu_3)
        # fourth conv block
        conv_4 = Conv2D(filters=128, kernel_size=KERNEL_SIZE,
                        strides=STRIDE, name='conv_4')(drop_3)
        relu_4 = LeakyReLU(LEAKY_ALPHA, name=f'relu_4')(conv_4)
        drop_4 = Dropout(rate=DROPOUT, name=f'drop_4')(relu_4)
        # dense net
        flat = Flatten(name='flat')(drop_4)
        outputs = Dense(units=1, activation='sigmoid', name='outputs')(flat)
        # model saved
        model = Model(inputs=img_in, outputs=outputs)
        self.discriminatorStruct = model
        return True

    def build_describer(self):
        """
        Builds structure for the describer model, which uses convolutional
        and dense networks to convert an image into EMBEDDING_DIM vector.
        """
        # cache params
        IMG_SHAPE = self.IMG_SHAPE
        DROPOUT = self.DROPOUT
        KERNEL_SIZE = self.KERNEL_SIZE
        STRIDE = self.STRIDE
        LEAKY_ALPHA = self.LEAKY_ALPHA
        # first conv block
        img_in = Input(shape=IMG_SHAPE, name='img_in')
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
        conv_3 = Conv2D(filters=64, kernel_size=KERNEL_SIZE,
                        strides=STRIDE, name='conv_3')(drop_2)
        relu_3 = LeakyReLU(LEAKY_ALPHA, name=f'relu_3')(conv_3)
        drop_3 = Dropout(rate=DROPOUT, name=f'drop_3')(relu_3)
        # fourth conv block
        conv_4 = Conv2D(filters=128, kernel_size=KERNEL_SIZE,
                        strides=STRIDE, name='conv_4')(drop_3)
        relu_4 = LeakyReLU(LEAKY_ALPHA, name=f'relu_4')(conv_4)
        drop_4 = Dropout(rate=DROPOUT, name=f'drop_4')(relu_4)
        # dense network
        flat = Flatten(name='flat')(drop_4)
        outputs = Dense(units=self.EMBEDDING_DIM, activation='sigmoid',
                        name='outputs')(flat)
        # model saved
        model = Model(inputs=img_in, outputs=outputs)
        self.describerStruct = model
        return True

    def compile_discriminator(self, learningRate=0.1, decay=0.1):
        """ Compiles discriminator model """
        rmsOptimizer = RMSprop(lr=learningRate, decay=decay)
        binaryLoss = 'binary_crossentropy'
        discriminatorModel = self.discriminatorStruct
        discriminatorModel.compile(optimizer=rmsOptimizer, loss=binaryLoss,
                                metrics=['accuracy'])
        self.discriminatorModel = discriminatorModel
        return discriminatorModel

    def compile_describer(self, learningRate=0.1, decay=0.1):
        """ Compiles describer using custom distance_loss """
        rmsOptimizer = RMSprop(lr=learningRate, decay=decay)
        describerModel = self.describerStruct
        describerModel.compile(optimizer=rmsOptimizer,
                            loss=self.distance_loss(describerModel.layers[-1]))
        self.describerModel = describerModel
        return describerModel

    def compile_adversarial(self, learningRate=0.1, decay=0.1):
        """ Compiles adversarial model as generator -> discriminator """
        rmsOptimizer = RMSprop(lr=learningRate, decay=decay)
        binaryLoss = 'binary_crossentropy'
        # adversarial built by passing generator output through discriminator
        adversarialModel = Sequential()
        adversarialModel.add(self.generatorStruct)
        adversarialModel.add(self.discriminatorStruct)
        adversarialModel.compile(optimizer=rmsOptimizer, loss=binaryLoss,
                                metrics=['accuracy'])
        self.adversarialModel = adversarialModel
        return adversarialModel

    def compile_creative(self, learningRate=0.1, decay=0.1):
        """
        Compiles creative model as generator -> describer using distance_loss
        """
        rmsOptimizer = RMSprop(lr=learningRate, decay=decay)

        creativeModel = Sequential()
        creativeModel.add(self.generatorStruct)
        creativeModel.add(self.describerStruct)
        creativeModel.compile(optimizer=rmsOptimizer,
                        loss=self.distance_loss(creativeModel.layers[-1]))
        self.creativeModel = creativeModel
        return creativeModel

    def initialize_models(self):
        """ Top-level func compiles all models and sets initialized to True """
        assert (self.initizalized == None), 'model has already been built.'
        self.build_generator()
        self.build_discriminator()
        self.build_describer()
        self.compile_discriminator()
        self.compile_describer()
        self.compile_adversarial()
        self.compile_creative()
        self.initizalized = True
        return True

    ## IMAGE MANIPULATION ##
    def image_from_textVec(self, textVec):
        """ Uses generator model to generate image from BERT matrix """
        return self.generatorStruct.predict(textVec)

    ## TRAINING ##
    def train_models(self, ):
        """
        Train summarizer, generator, discriminator, describer, adversarial,
        and creative models on dataset with end-goal of text-to-image
        generation.
        """
        pass





x = Image_Generator(1,1,1)
x.initialize_models()
