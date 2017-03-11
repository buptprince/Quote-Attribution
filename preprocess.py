import pickle, os, re
import nltk

class Preprocess:

    def cleanFile(self, fname):
        script = pickle.load(open(fname, 'rb'))
        for i in xrange(len(script)):
            script[i][0] = self.cleanName(script[i][0])
            print self.cleanDial(script[i][1])

    def cleanName(self, name):
        return name.strip().lower()

    def cleanDial(self, d):
        d = d.strip().lower()
        d = re.sub(r'(\.\.+)|"|:|;', ' ', d)
        d = re.sub(r'\s*\([^)]*\)', '', d)
        d = re.sub(r'\s+', ' ', d)
        return d.strip()

    def cleanAllFiles(self):
        for f in os.listdir('data'):
            self.cleanFile(os.path.join("data", f))
            break

if __name__ == '__main__':
    Preprocess().cleanAllFiles()
