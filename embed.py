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

        self.patch = ["george", ["goerge", "georgge"]]

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
        root = self.config.cleanedRoot
        vecMatRoot = self.config.qVecMatPath
        for fname in os.listdir(root):
            pth = os.path.join(root, fname)
            mat = []
            with open(pth, 'rb') as f:
                for dial in pickle.load(f):
                    if len(dial) == 2 and isinstance(dial[1], list):
                        spk = None
                        if dial[0] in self.patch[1]:
                            spk = self.util.speakers.index(self.patch[0])
                        else:
                            spk = self.util.speakers.index(dial[0])
                        qVec = self.getQuoteVec(dial[1])
                        # qVec = np.random.rand(50)
                        if qVec == None:
                            continue
                        mat.append(np.append(np.array([spk]), qVec))
            mat = np.mat(mat)
            f_ = fname.split('.')[0]+".vecmat.bin"
            f_ = os.path.join(vecMatRoot, f_)

            with open(f_, 'wb') as f:
                pickle.dump(mat, f)
                print "[SUCCESS] ", f_, mat.shape



if __name__ == '__main__':
    obj = Embed()
    obj.saveQuoteVec()
    # print obj.getQuoteVec(["hello", "my", "name", "is", "sayan"])
