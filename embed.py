'''
 Word Embedding Class
 Transforms given sentence into 50d word vector.
 If,
 Q = {W1, W2, W3, ... Wn}
 Wn = [a1, a2, a3, ... an]
 Qvec = Simple mean of Q along axis 0

 Methods:
    loadData(): open the pretrained GloVe files and create Gensim model
    getQuoteVec(): convert the sentence into 50d vector
'''

import numpy as np
from config import Config
from util import Util
import os, pickle, gensim

class Embed:
    def __init__(self):
        self.config = Config()
        self.loadData()
        self.util = Util()

    def loadData(self):
        gensim_file = os.path.join(self.config.wordVecRoot, self.config.wordVecModelPath)
        self.wordVec = gensim.models.KeyedVectors.load_word2vec_format(gensim_file, binary=False)

    def getQuoteVec(self, sent):
        qVec = []
        for tok in sent:
            if tok in self.wordVec:
                qVec.append(self.wordVec[tok])

        if not len(qVec):
            return None
        else:
            return np.mean(np.array(qVec), axis=0)

    def saveQuoteVec(self):
        print "Saving Quote Vector Matrix"
        pth = self.config.cleanedData
        vecMatPath = self.config.qVecMatPath
        with open(pth, 'rb') as f:
            mat = []
            for dial in pickle.load(f):
                spk = self.util.speakers.index(dial[0])
                qVec = self.getQuoteVec(dial[1])
                # qVec = np.random.rand(50)
                if qVec is None:
                    continue
                mat.append(np.append(np.array([spk]), qVec))
            mat = np.array(mat)
            with open(vecMatPath, 'wb') as f:
                pickle.dump(mat, f)
                print "[SUCCESS] ", vecMatPath, mat.shape

if __name__ == '__main__':
    obj = Embed()
    obj.saveQuoteVec()
    # print obj.getQuoteVec(["hello", "my", "name", "is", "sayan"])
