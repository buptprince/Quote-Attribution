from bs4 import BeautifulSoup as bs
import urllib2, pickle

class Scrape:

    def __init__(self):
        self.url = "http://www.imsdb.com/TV/Seinfeld.html"
        self.root = "http://www.imsdb.com"
        self.extractLinks()

    def extractLinks(self):
        with open("cache/%s" % self.url.split('/')[-1], "r") as f:
            soup = bs(f.read())
            mdiv = soup.find_all(valign="top")[1]
            mdiv = mdiv.find(valign="top")

            series = []
            i = -1
            for child in mdiv.descendants:
                if child.name == "h2":
                    i =  int(child.text.split()[-1])-1
                    series.append([])
                    continue
                if child.name == 'p':
                    series[i].append(self.root+child.a['href'])
        with open("cache/links.pkl", "wb") as f:
            pickle.dump(series, f)
            del series, soup


    def cache(self):
        html = urllib2.urlopen(self.url).read()
        with open("cache/%s" % self.url.split('/')[-1], "w") as f:
            f.write(html)

if __name__ == '__main__':
    Scrape()
