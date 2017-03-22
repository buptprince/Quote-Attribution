'''
 Preprocessing Class
 Clean and process the raw files into <sess>.cleaned.bin

 Methods:
    cleanFile(): for a given file name it cleans and process data
    cleanName(): process speaker names
    cleanDial(): clean and process the dialogue
    tokenizeQuote(): tokenize the quote by words
'''

import pickle, os, re
import nltk

class Preprocess:

    def cleanFile(self, fname):
        script = pickle.load(open(os.path.join("data", fname), 'rb'))
        for i in xrange(len(script)):
            script[i][0] = self.cleanName(script[i][0])
            script[i][1] = self.cleanDial(script[i][1])
            script[i][1] = self.tokenizeQuote(script[i][1])

            # Check for 'jerry and john' and split it 2 unique quotes
            if re.match(r'(.+) and (.*)', script[i][0]):
                actors = script[i][0].split(" and ")

                script[i][0] = actors[0]
                script.insert(i, [actors[1], script[i][1]])
        fname = fname.split('.')
        fname.insert(1, 'clean')
        fname = ".".join(fname)
        pickle.dump(script, open(os.path.join("data", "cleaned", fname), 'wb'))
        del script
        print "[SUCCESS] Cleaned", fname

    def cleanName(self, name):
        return name.strip().lower()

    def cleanDial(self, d):
        if not isinstance(d, list):
            d = d.strip().lower()
            d = re.sub(r'(\.\.+)|"|:|;', ' ', d) # Remove Multiple Spaces & Special ch.
            d = re.sub(r'\s*\([^)]*\)', '', d) # Remove bracket and it's content
            d = re.sub(r'\s+', ' ', d) # Remove Multiple Spaces
            return d.strip()
        else:
            return d

    def cleanAllFiles(self):
        for f in os.listdir('data'):
            if f == "cleaned":
                continue
            self.cleanFile(f)

    def tokenizeQuote(self, quote):
        if not isinstance(quote, list):
            return nltk.word_tokenize(quote)
        else:
            return quote

if __name__ == '__main__':
    Preprocess().cleanAllFiles()
