'''
 Configuration Class
 Contains all the tuning parameters for all the models
 along with the word embedding configs

 Attributes:
    wordVecRoot: root dir of glove pretrained dataset
    wordVecPath: pretrained word vector path
    wordDim: Dimension of word vector
    nToken: number of words
'''

class Config:
    def __init__(self):
        self.urlRoot = "http://www.imsdb.com"
        self.url = "http://www.imsdb.com/TV/Seinfeld.html"

        self.wordVecRoot = "./glove"
        self.wordVecPath = "glove.6B.50d.txt"
        self.wordVecModelPath = "glove.6B.50d.model.txt"
        self.wordDim = int(self.wordVecPath.split('.')[2][:-1])
        self.nTokens = int(self.wordVecPath.split('.')[1][:-1])

        self.cleanedRoot = "./data/cleaned"
        self.cachedLinks = "cache/links.pkl"

        self.qVecMatPath = "./data/vec"

        # Parameter for Machine Learning Models
        # MLP Model
        self.mlp = {
            'nHidden': 220,
            'alpha': 0.2,
            'stddev': 0.5,
            'train': 0.70,
            'epochs': 2000,
            'disp': 50,
            'modelPath': './bin/mlp.ckpl'
        }

        # RNN Model
        self.rnn = {
            'basic': {
                'numStep': 10,
                'stateSize': 12,
                'alpha': 0.2,
                'stddev': 0.3,
                'train': 0.7,
                'epochs': 5000,
                'disp': 50,
                'modelPath': './bin/rnn.basic.ckpl'
            },
            'LSTM': {
                'numStep': 5,
                'stateSize': 140,
                'alpha': 0.2,
                'stddev': 0.5,
                'train': 0.7,
                'epochs': 5000,
                'disp': 50,
                'modelPath': './bin/rnn.lstm.ckpl'
            },
            'GRU': {
                'numStep': 4,
                'stateSize': 42,
                'alpha': 0.2,
                'stddev': 0.5,
                'train': 0.7,
                'epochs': 2500,
                'disp': 50,
                'modelPath': './bin/rnn.gru.ckpl'
            }
        }
