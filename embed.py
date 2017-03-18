import numpy as np
from config import Config
import os, pickle, gensim

class Embed:
    def __init__(self):
        self.config = Config()
        self.loadData()

    def loadData(self):
        gensim_file = os.path.join(self.config.wordVecRoot, self.config.wordVecModelPath)
        self.wordVec = gensim.models.KeyedVectors.load_word2vec_format(gensim_file, binary=False)

    def getQuoteVec(self, sent):
        qVec = []
        for tok in sent:
            qVec.append(self.wordVec[tok])
        return np.array(qVec)

if __name__ == '__main__':
    obj = Embed()
    obj.getQuoteVec(["hello", "my", "name", "is", "sayan"])
