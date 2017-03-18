
'''
    Contains all the tuning parameters for all the models
    along with the word embedding configs
'''

class Config:
    def __init__(self):
        self.wordVecRoot = "./glove"
        self.wordVecPath = "glove.6B.50d.txt"
        self.wordVecModelPath = "glove.6B.50d.model.txt"
        self.wordDim = int(self.wordVecPath.split('.')[2][:-1])
        self.nTokens = int(self.wordVecPath.split('.')[1][:-1])

        self.cleanedRoot = "./data/cleaned"


        # self.embedIndRoot = "./embed"
        # self.embedIndPkl = "emb.6B.50d.pkl"
