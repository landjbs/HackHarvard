from keras.models import Model, Sequential
from keras.layers import Dense, GRU, LSTM, Bidirectional, Input

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

    def build_summarizer(self, verbose=False):
        """ Builds architecture of summarizer """
        assert (self.summarizerStructure==None), 'summarizer already built'
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
        if verbose:
            print(summarizerStructure.summary())
        self.summarizerStructure = summarizerStructure
        return True



x = Summarizer_Model(100)
x.build_summarizer(verbose=True)
