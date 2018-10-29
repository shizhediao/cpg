#! /usr/bin/env python
#-*- coding:utf-8 -*-
#训练字级别的词向量

from utils import *
from vocab import get_vocab, VOCAB_SIZE
from corpus import get_all_corpus
from gensim import models
from numpy import zeros, array
from numpy.random import uniform
import re
from math import exp

_w2v_path = os.path.join(data_dir, 'word2vec.npy')
_w2v_text = os.path.join(data_dir, 'word2vec.txt')
_w2v_model = os.path.join(data_dir, 'word2vec.model')

def _gen_embedding(ndim):
    print ("Generating %d-dim word embedding ..." %ndim)
    int2ch, ch2int = get_vocab()
    ch_lists = []
    corpus = get_all_corpus()
    for idx, poem in enumerate(corpus):
        if 'paragraphs' in poem:
            #ss = []
            for sentence in poem['paragraphs']:
                #print(re.split('[，。；？]', sentence))
                for s in re.split('[，。；？]', sentence):
                    if s != '':
                        ch_lists.append(s)
            #ch_lists.append(ss)
            #print(ch_lists)
            #print(ss)
        if 0 == (idx+1)%10000:
            print ("[Word2Vec] %d/%d poems have been processed." %(idx+1, len(corpus)))
    print ("Hold on. This may take some time ...")
    model = models.Word2Vec(ch_lists, size=300, window=5, min_count=5, iter=100)
    embedding = uniform(-1.0, 1.0, [VOCAB_SIZE, ndim])
    for idx, ch in enumerate(int2ch):
        if ch in model.wv:
            embedding[idx,:] = model.wv[ch]
    embedding[0,:] = zeros(ndim)
    embedding[VOCAB_SIZE-1,:] = zeros(ndim)
    print(embedding)
    model.save(_w2v_model)
    model.wv.save_word2vec_format(_w2v_text, binary=False)
    np.save(_w2v_path, embedding)
    print ("Word embedding is saved.")

def get_word_embedding(ndim):
    if not os.path.exists(_w2v_path):
        _gen_embedding(ndim)
    embedding = np.load(_w2v_path)
    return embedding

if __name__ == '__main__':
    embedding = get_word_embedding(300)
    print ("Size of embedding: (%d, %d)" %embedding.shape)


