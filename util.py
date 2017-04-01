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
import os, pickle
from preprocess import Preprocess

class Util:
    def __init__(self):
        self.config = Config()
        self.preprocess = Preprocess()
        self.speakers = []

        self.getSpeakers()

    def getSpeakers(self):
        root = self.config.cleanedRoot
        for f in os.listdir(root):
            pth = os.path.join(root, f)
            with open(pth, 'rb') as f:
                self.addSpeaker(pickle.load(f))
        return self.speakers

    def addSpeaker(self, sess):
        for line in sess:
            name = self.preprocess.cleanName(line[0])
            if not name in self.speakers:
                self.speakers.append(name)

if __name__ == '__main__':
    obj = Util()
