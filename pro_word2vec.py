#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from vocab import get_vocab, VOCAB_SIZE
from corpus import get_all_corpus
from gensim import models
from numpy import zeros, array
from numpy.random import uniform
import re

_w2v_path = os.path.join(data_dir, 'pro_word2vec.npy')
_w2v_text = os.path.join(data_dir, 'word2vec.txt')
_w2v_model = os.path.join(data_dir, 'word2vec.model')
pro_path = os.path.join(data_dir, 'projection_orig1_projected.txt')

int2ch, ch2int = get_vocab()
ndim = 300
model = models.KeyedVectors.load_word2vec_format(pro_path, binary=False)
embedding = uniform(-1.0, 1.0, [VOCAB_SIZE, ndim])
for idx, ch in enumerate(int2ch):
    if ch in model.wv:
        embedding[idx,:] = model.wv[ch]
embedding[0,:] = zeros(ndim)
embedding[VOCAB_SIZE-1,:] = zeros(ndim)
print(embedding)
model.save(_w2v_model)
np.save(_w2v_path, embedding)
print ("Word embedding is saved.")
