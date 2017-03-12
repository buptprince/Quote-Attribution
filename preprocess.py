import pickle, os, re
import nltk

class Preprocess:

    def cleanFile(self, fname):
        script = pickle.load(open(fname, 'rb'))
        for i in xrange(len(script)):
            script[i][0] = self.cleanName(script[i][0])
            script[i][1] = self.cleanDial(script[i][1])

            if re.match(r'(.+) and (.*)', script[i][0]):
                actors = script[i][0].split(" and ")

                script[i][0] = actors[0]
                script.insert(i, [actors[1], script[i][1]])

        fname = fname.split('.')
        fname.insert(1, 'clean')
        fname = ".".join(fname)
        pickle.dump(script, open(fname, 'wb'))
        del script
        print "[SUCCESS] Cleaned", fname

        
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

    def tokenizeFile(self, fname):
        pass

if __name__ == '__main__':
    Preprocess().cleanAllFiles()
