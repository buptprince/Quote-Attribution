import numpy as np
from config import Config
import os, pickle

class Embed:
    def __init__(self):
        self.config = Config()

        if not os.path.exists(os.path.join(self.config.embedIndRoot, self.config.embedIndPkl)):
            self.getEmbedInd()

    def getEmbedInd(self):
        embeddings_index = {}

        print "Opening", self.config.wordVecPath
        f = open(os.path.join(self.config.wordVecRoot, self.config.wordVecPath))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))
        with open(os.path.join(self.config.embedIndRoot, self.config.embedIndPkl), 'wb') as f:
            pickle.dump(embeddings_index, f)
            print "Pickled Embedding Matrix as", self.config.embedIndPkl

    def _loadEmbInd(self):
        pass
        
    def getQVec(self):
        pass


if __name__ == '__main__':
    obj = Embed()
    obj.getEmbedInd()
