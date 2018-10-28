#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from segment import Segmenter
from vocab import *
from quatrains import get_quatrains
from rank_words import get_word_ranks
import numpy as np
import shutil
import random


train_path = os.path.join(data_dir, 'train.txt')

kw_train_path = os.path.join(data_dir, 'kw_train.txt')

def _gen_train_data():
    segmenter = Segmenter()
    poems = get_quatrains()
    random.shuffle(poems)
    ranks = get_word_ranks()
    print ("Generating training data ...")
    data = []
    kw_data = []
    for idx, sentence in enumerate(poems):
        rows = []
        kw_row = []
        rows.append([sentence])
        segs = list(filter(lambda seg: seg in ranks, segmenter.segment(sentence)))
        if 0 == len(segs):
            continue
        keyword = reduce(lambda x,y: x if ranks[x] < ranks[y] else y, segs)
        kw_row.append(keyword)
        rows[-1].append(keyword)
        data.extend(rows)
        kw_data.append(kw_row)
        if 0 == (idx+1)%10000:
            print ("[Training Data] %d/%d sentences are processed." %(idx+1, len(poems)))
    with codecs.open(train_path, 'w', 'utf-8') as fout:
        for row in data:
            fout.write('\t'.join(row)+'\n')
    with codecs.open(kw_train_path, 'w', 'utf-8') as fout:
        for kw_row in kw_data:
            fout.write('\t'.join(kw_row)+'\n')
    print ("Training data is generated.")


def get_train_data():
    if not os.path.exists(train_path):
        _gen_train_data()
    data = []
    with codecs.open(train_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            toks = line.strip().split('\t')
            data.append({'sentence':toks[0], 'keyword':toks[1]})
            line = fin.readline()
    return data

def get_test_data():
    if not os.path.exists(train_path):
        _gen_train_data()
    data = []
    with codecs.open('test.txt', 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            toks = line.strip().split('\t')
            data.append({'sentence':toks[0], 'keyword':toks[1]})
            line = fin.readline()
    return data

def get_kw_train_data():
    if not os.path.exists(kw_train_path):
        _gen_train_data()
    data = []
    with codecs.open(kw_train_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            data.append(line.strip().split('\t'))
            line = fin.readline()
    return data

def _gen_dataset():
    if not os.path.exists(train_path):
        _gen_train_data()
    _, ch2int = get_vocab()
    data_s_7_t = []
    data_s_7_v = []
    data_kw_7_t = []
    data_kw_7_v = []
    i = 0
    ss = []
    kws = []
    with codecs.open(train_path, 'r', 'utf-8') as fin:
        while True:
            if i%4 == 0:
                ss = []
                kws = []
            i = i+1
            line = fin.readline()
            if not line:
                break
            else:
                toks = line.strip().split('\t')
                s = [0]
                s.extend([ch2int[ch] for ch in toks[0]])
                kw = [ch2int[ch] for ch in toks[1]]
                #if len(s) < 8:
                #    s.extend([VOCAB_SIZE-1,VOCAB_SIZE-1])
                while len(kw) < 3:
                    kw.extend([VOCAB_SIZE-1])
                ss.append(s)
                kws.append(kw)
                if len(s) > 7 and i%4 == 0:
                    if random.random() < 0.8:
                        data_s_7_t.extend(ss)
                        data_kw_7_t.extend(kws)
                    else:
                        data_s_7_v.extend(ss)
                        data_kw_7_v.extend(kws)
    print(len(data_s_7_t))
    print(len(data_kw_7_t))
    print(len(data_s_7_v))
    print(len(data_kw_7_v))
    np.save('sentence7t', data_s_7_t)
    np.save('sentence7v', data_s_7_v)
    np.save('keyword7t', data_kw_7_t)
    np.save('keyword7v', data_kw_7_v)

def find_repeat_word():
    word_list1 = set()
    word_list2 = set()
    with codecs.open(train_path, 'r', 'utf-8') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            tok = line.strip().split('\t')[0]
            for idx, w1 in enumerate(tok):
                for _, w2 in enumerate(tok[idx+1:]):
                    #print(idx, w1, _, w2)
                    if w1 == w2:
                        word_list1.add(w1)
                        if _ == 0:
                            word_list2.add(w2)
                        
    #print(word_list1)
    #print(len(word_list1))
    #print(word_list2)
    #print(len(word_list2))
    return word_list1, word_list2

if __name__ == '__main__':
    train_data = get_train_data()
    print ("Size of the training data: %d" %len(train_data))
    kw_train_data = get_kw_train_data()
    print ("Size of the keyword training data: %d" %len(kw_train_data))
    _gen_dataset()
    #find_repeat_word()

