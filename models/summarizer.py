from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Bidirectional, Input

class Summarizer_Model():
    """
    Model to summarize text by modifying BERT vec using LSTM or Dense Net
    """
    def __init__(self):
        # length of bert vectors
        self.BERT_DIM               = 1024
        self.LATENT_DIM             = 1024
        # architecture of summarizer model
        self.summarizerStructure    = None
        # compiled summarizer model
        self.summarizerModel        = None

    def build_summarizer(self):
        """ Builds architecture of summarizer """
        inputs = Input(shape=(self.BERT_DIM, ), name='encodings')
        lstm_in = LSTM(units=self.LATENT_DIM, activation='relu', )

inputs = keras.layers.Input(shape=[404, 768], name='encodings')
    # gru_out = keras.layers.Bidirectional(keras.layers.GRU(units=768, return_sequences=True))(inputs)
    lstm_in = keras.layers.LSTM(units=768, return_sequences=True)(inputs)
    # dense = keras.layers.TimeDistributed(keras.layers.Dense(units=1, activation='softmax'))(dense)
    flat = keras.layers.Flatten()(lstm_in)
    output = keras.layers.Dense(units=404, activation='softmax')(flat)
