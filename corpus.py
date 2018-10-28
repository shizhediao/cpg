#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *

def get_all_corpus():
    corpus = []
    dirname = 'poetry'
    for fname in os.listdir(dirname):
        #print(fname)
        with codecs.open(os.path.join(dirname, fname),'r','utf-8') as fin:
            data = json.load(fin)
        corpus.extend(data)
    return corpus


if __name__ == '__main__':
    corpus = get_all_corpus()
    print ("Size of the entire corpus: %d" % len(corpus))

