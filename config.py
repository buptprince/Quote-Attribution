
'''
    Contains all the tuning parameters for all the models
    along with the word embedding configs
'''

class Config:
    def __init__(self):
        self.wordVecPath = "glove.840B.300d.txt"
        self.wordDim = int(self.wordVecPath.split('.')[2][:-1])
