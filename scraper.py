from bs4 import BeautifulSoup as bs
import urllib2, pickle, re

class Scrape:

    def __init__(self):
        self.url = "http://www.imsdb.com/TV/Seinfeld.html"
        self.root = "http://www.imsdb.com"

        # self.extractLinks()
        self.scrapeLinks()

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

    def scrapeLinks(self):
        with open("cache/links.pkl", "rb") as f:
            series =  pickle.load(f)
            for season in series:
                for ep in season:
                    self.scrapePage(ep)
                    break
                break


    def scrapePage(self, link):
        link = urllib2.quote(link).replace("http%3A", "http:")
        html = urllib2.urlopen(link).read()
        soup = bs(html)

        reg = re.compile(r'Read \"[\s\S]*\" Script')
        elements = [e for e in soup.find_all('a') if reg.match(e.text)][0]
        self.extractScript(self.root+urllib2.quote(elements['href']))


    def extractScript(self, link):
        html = urllib2.urlopen(link).read()
        soup = bs(html)

        pre = soup.find('td', class_="scrtext").find('pre')
        counter = 0
        for c in pre.strings:
            c = re.sub(r'(\s\s+)|(\n)', ' ', c)

            if re.match(r'\s[A-Z]+\s$|\s[A-Z]+\sAND\s[A-Z]+\s$', c):
                print c
            # print len(c), "++", c, "--"


    def cache(self):
        html = urllib2.urlopen(self.url).read()
        with open("cache/%s" % self.url.split('/')[-1], "w") as f:
            f.write(html)

if __name__ == '__main__':
    Scrape()
