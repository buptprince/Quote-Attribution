'''
 Utility Class
 Contains utiliy methods used all over the project.

 Attributes:
    speakers: list of unique speaker indexed in an array

 Methods:
    getSpeaker(): opens every file and add speakers
    addSpeaker(): adds speakers from a particular file
'''

from config import Config
from preprocess import Preprocess
import os, pickle
import numpy as np

class Util:
    def __init__(self):
        self.config = Config()
        self.preprocess = Preprocess()
        self.speakers = []
        self.nSpeakers = 0

        self.patch = ["george", ["goerge", "georgge"]]
        self.mainCharacters = [
            "george",
            "jerry",
            "kramer",
            "laura"
        ]

        self.getSpeakers()

    def getSpeakers(self):
        root = self.config.cleanedRoot
        for f in os.listdir(root):
            pth = os.path.join(root, f)
            with open(pth, 'rb') as f:
                self.addSpeaker(pickle.load(f))
        self.nSpeakers = len(self.speakers)
        return self.speakers

    def addSpeaker(self, sess):
        for line in sess:
            name = self.preprocess.cleanName(line[0])
            if name in self.patch[1]:
                name = self.patch[0]
            if not name in self.speakers:
                self.speakers.append(name)

    def loadData(self, redChars = False):
        dataPath = self.config.qVecMatPath
        X = None

        for fname in os.listdir(dataPath):
            with open(os.path.join(dataPath, fname), 'rb') as f:
                mat = pickle.load(f)
                if X is None:
                    X = mat
                    continue
                X = np.append(X, mat, axis=0)

        X= np.array(X)
        if redChars:
            X = self.reduceCharacters(X)
        Y = np.array(X[:, 0], dtype=int)
        Y = np.eye(self.nSpeakers)[Y]
        X = X[:, 1:]
        return X, Y

    def reduceCharacters(self, mat):
        for char in self.mainCharacters:
            Oi = self.speakers.index(char)
            Ni = (self.mainCharacters.index(char)+1)*(-1) # Negative index to prevent updating updated values again

            mat[np.where(mat[:,0] == Oi),0] = Ni
        # Bin speakers not in mainCharacters list to 'others'
        mat[np.where(mat[:,0] >= 0), 0] = len(self.mainCharacters)+1-1
        # Convert negative index to positive form
        mat[np.where(mat[:,0] < 0), 0] = mat[np.where(mat[:,0] < 0), 0]*(-1)-1
        self.nSpeakers = len(self.mainCharacters)+1
        return mat

if __name__ == '__main__':
    obj = Util()
    obj.loadData(redChars=True)
    print obj.speakers
