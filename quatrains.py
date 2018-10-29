#! /usr/bin/env python
#-*- coding:utf-8 -*-
#从corpus中提取出7个字且由字典中的字组成的诗

from utils import *
from corpus import get_all_corpus
from vocab import get_vocab
import re

def get_quatrains():
    _, ch2int = get_vocab()
    corpus = get_all_corpus()
    quatrains = []
    for idx, poem in enumerate(corpus):
        if 'paragraphs' in poem:
            for sentence in poem['paragraphs']:
                for s in re.split('[，。；？]', sentence):
                    if s != '' and len(s) == 7:
                        flag = True
                        for ch in s:
                            if ch not in ch2int:
                                flag = False
                                break
                        #if flag:
                            #print(s)
                        quatrains.append(s)
        if idx%10000==0:
            print("%d / %d poems have been done" % (idx,len(corpus)))
    return quatrains


if __name__ == '__main__':
    quatrains = get_quatrains()
    #print(quatrains)
    print ("Size of quatrains: %d" % len(quatrains))

