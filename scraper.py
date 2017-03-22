'''
 Scraper Class
 Scrapes the url and stores the raw data

 Attributes:
    process: stores the data for current in process episode

 Methods:
    extractLinks(): extract all the episode links from Config.url
    scrapeLink(): for every link scrape the page
    scrapePage(): scrape the link for the givel url
    extractScript(): extract the scripts from the page
'''


from bs4 import BeautifulSoup as bs
from config import Config
import urllib2, pickle, re, os

class Scraper:

    def __init__(self):
        self.configs = Config()
        self.process = {
            'season': None,
            'episode': None
        }

        # self.extractLinks()
        self.scrapeLinks()

    def extractLinks(self):
        with open("cache/%s" % self.configs.url.split('/')[-1], "r") as f:
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
                    series[i].append(self.configs.urlRoot+child.a['href'])
        with open(self.configs.cachedLinks, "wb") as f:
            pickle.dump(series, f)
            del series, soup

    def scrapeLinks(self):
        with open(self.configs.cachedLinks, "rb") as f:
            series =  pickle.load(f)
            for season in series:
                self.process['season'] = series.index(season)+1
                for ep in season:
                    self.scrapePage(ep)
                    self.process['episode'] = season.index(ep)+1

    def scrapePage(self, link):
        link = urllib2.quote(link).replace("http%3A", "http:")
        html = urllib2.urlopen(link).read()
        soup = bs(html)

        reg = re.compile(r'Read \"[\s\S]*\" Script')
        elements = [e for e in soup.find_all('a') if reg.match(e.text)][0]
        self.extractScript(self.configs.urlRoot+urllib2.quote(elements['href']))


    def extractScript(self, link):
        html = urllib2.urlopen(link).read()
        soup = bs(html)

        pre = soup.find('td', class_="scrtext").find('pre')
        tscript = []
        speaker, dialogue = None, None
        for c in pre.strings:
            c = re.sub(r'(\s\s+)|(\n)', ' ', c)

            if re.match(r'\s[A-Z]+\s$|\s[A-Z]+\sAND\s[A-Z]+\s$', c):
                if speaker:
                    tscript.append([speaker, dialogue])
                    dialogue = None
                speaker = c
            else:
                if dialogue:
                    dialogue += c
                else:
                    dialogue = c

        with open("data/%s.bin"%str(len(os.listdir("./data"))), "wb") as f:
            pickle.dump(tscript, f)
            print "[SUCCESS] tScript Season.", self.process['season'], "Episode", self.process['episode']


    def cache(self):
        html = urllib2.urlopen(self.configs.url).read()
        with open("cache/%s" % self.configs.url.split('/')[-1], "w") as f:
            f.write(html)

if __name__ == '__main__':
    Scraper()
