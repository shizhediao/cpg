#! /usr/bin/env python
# -*- coding:utf-8 -*-

from data_utils import *
from plan import Planner
from generate_new import Generator

if __name__ == '__main__':
    planner = Planner()
    generator = Generator()
    while True:
        line = input('Input Text:\t').strip()
        if line.lower() == 'quit' or line.lower() == 'exit':
            break
        elif len(line) > 0:
            keywords = planner.plan(line)
            #keywords = line.strip().split()
            print ("Keywords:\t",)
            for word in keywords:
                print (word,)
            print ('\n')
            print ("Poem Generated:\n")
            generator.ya = 0
            generator.yalist = []
            sentences = generator.generate(keywords)
            print ('\t'+sentences[0]+u'，'+sentences[1]+u'。')
            print ('\t'+sentences[2]+u'，'+sentences[3]+u'。')
            print()
