from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Bidirectional, Input

class Summarizer_Model():
    """
    Model to summarize text by modifying BERT vec using LSTM or Dense Net
    """
    def __init__(self, maxLen):
        # length of bert vectors
        self.BERT_DIM               = 1024
        self.LATENT_DIM             = 1024
        self.MAX_CAPTION_LEN        = maxLen
        # architecture of summarizer model
        self.summarizerStructure    = None
        # compiled summarizer model
        self.summarizerModel        = None

    def build_summarizer(self):
        """ Builds architecture of summarizer """
        # matix of word encodings
        word_inputs = Input(shape=(self.BERT_DIM, self.MAX_CAPTION_LEN),
                            name='word_encodings')
        # vector of cls
        cls_input = Input(shape=(self.BERT_DIM, ), name='cls_input')
        # define structure of bidirectional lstm
        lstm_in = Bidirectional(LSTM(units=self.LATENT_DIM, activation='relu',
                                    return_state=return_state))
        # get outputs of lstm run over word_inputs with cls initial state
        lstm_out = Bidirectional(lstm_in(word_inputs, intial_state=cls_input))


inputs = keras.layers.Input(shape=[404, 768], name='encodings')
    # gru_out = keras.layers.Bidirectional(keras.layers.GRU(units=768, return_sequences=True))(inputs)
    lstm_in = keras.layers.LSTM(units=768, return_sequences=True)(inputs)
    # dense = keras.layers.TimeDistributed(keras.layers.Dense(units=1, activation='softmax'))(dense)
    flat = keras.layers.Flatten()(lstm_in)
    output = keras.layers.Dense(units=404, activation='softmax')(flat)
