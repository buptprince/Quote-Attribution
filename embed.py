import numpy as np
from config import Config
import os, pickle

class Embed:
    def __init__(self):
        self.config = Config()

        if not os.path.exists(os.path.join(self.config.embedIndRoot, self.config.embedIndPkl)):
            self.getEmbedInd()

    def getEmbedInd(self):
        embedInd = {}

        print "Opening", self.config.wordVecPath
        f = open(os.path.join(self.config.wordVecRoot, self.config.wordVecPath))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedInd[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embedInd))
        with open(os.path.join(self.config.embedIndRoot, self.config.embedIndPkl), 'wb') as f:
            pickle.dump(embedInd, f)
            print "Pickled Embedding Matrix as", self.config.embedIndPkl

    def _loadEmbInd(self):
        print "Loading Word Embedding Index"
        with open(os.path.join(self.config.embedIndRoot, self.config.embedIndPkl), 'rb') as f:
            return pickle.load(f)

    def getQVec(self, d):
        emdedInd = self._loadEmbInd()


if __name__ == '__main__':
    obj = Embed()
    obj.getQVec(["hello", "my", "name", "is"])
