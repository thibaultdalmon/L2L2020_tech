import json
import logging
import os
import urllib

import bs4 as bs
import numpy as np


class Scraper:
    def __init__(self, args):
        self.train_dir = args.train_dir
        self.val_dir = args.val_dir
        self.test_dir = args.test_dir
        self.labels = args.labels
        self.multi_label_queries = args.queries

    def build_dataset(self):
        for single_label_queries in self.multi_label_queries:

            label = single_label_queries[0]
            total_images = 0

            for query in single_label_queries:

                query = query.split()
                query = '+'.join(query)
                url = "http://www.bing.com/images/search?q=" + query + "&FORM=HDRSC2"
                header = {
                    'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
                soup = self.get_soup(url, header)

                ActualImages = []
                for a in soup.find_all("a", {"class": "iusc"}):
                    m = json.loads(a["m"])
                    turl = m["turl"]
                    murl = m["murl"]

                    image_name = urllib.parse.urlsplit(murl).path.split("/")[-1]

                    ActualImages.append((image_name, turl, murl))

                total_images += len(ActualImages)

                for i, (image_name, turl, murl) in enumerate(ActualImages):

                    DIR, type = self.saveTo(label)
                    if not os.path.exists(DIR):
                        os.mkdir(DIR)

                    DIR = os.path.join(DIR)
                    if not os.path.exists(DIR):
                        os.mkdir(DIR)
                    try:
                        raw_img = urllib.request.urlopen(turl).read()
                        f = open(os.path.join(DIR, image_name), 'wb')
                        f.write(raw_img)
                        f.close()
                    except Exception as e:
                        logging.info("could not load : " + image_name)
                        logging.info(e)

            logging.info('Done for ' + label)

            logging.info("Total images : " + str(total_images))

        logging.info("All done")

    def get_soup(self, url, header):
        return bs.BeautifulSoup(urllib.request.urlopen(
            urllib.request.Request(url, headers=header)),
            'html.parser')

    def saveTo(self, label):
        a = np.random.random()
        if (a <= 0.7):
            return self.train_dir + label, 'train'
        elif (a > 0.97):
            return self.test_dir + 'mixed', 'test'
        else:
            return self.val_dir + label, 'val'
