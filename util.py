from config import Config
import os, pickle

class Util:
    def __init__(self):
        self.config = Config()
        self.speakers = []

    def getSpeakers(self):
        root = self.config.cleanedRoot
        for f in os.listdir(root):
            pth = os.path.join(root, f)
            with open(pth, 'rb') as f:
                self.addSpeaker(pickle.load(f))
            break

    def addSpeaker(self, sess):
        pass


if __name__ == '__main__':
    obj = Util()
    obj.getSpeakers()
