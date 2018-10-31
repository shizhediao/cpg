#! /usr/bin/env python
#-*- coding:utf-8 -*-
#训练词级别的词向量，最新的提关键词并扩充

from utils import *
from segment import Segmenter
from quatrains import get_quatrains
from rank_word import get_stopwords
from vocab import get_vocab
from data_utils import *
from rank_word import get_word_rank
import jieba
from gensim import models
from random import shuffle, random, randint
from functools import cmp_to_key
import operator

_model_path = os.path.join(data_dir, "word2vec.txt")
kw_model_path = os.path.join(data_dir, 'kw_model.bin')

class Planner:

    def __init__(self):
        self.rank = get_word_rank()
        self.ranks = get_word_ranks()
        self.stop_words = get_stopwords()
        self.model = models.KeyedVectors.load_word2vec_format(_model_path, binary=False)
        if os.path.exists(kw_model_path):
            self.kw_model = models.Word2Vec.load(kw_model_path)
        else:
            self._train()
        self.int2ch, self.ch2int = get_vocab()

    def _train(self):
        print ("Start training Word2Vec for planner ...")
        quatrains = get_quatrains()
        segmenter = Segmenter()
        seg_lists = []
        for idx, sentence in enumerate(quatrains):
            seg_list = []
            seg_list.extend(filter(lambda seg: seg in self.ranks,
                            segmenter.segment(sentence)))
            seg_lists.append(seg_list)
            if 0 == (idx+1)%10000:
                print ("[Plan Word2Vec] %d/%d sentences has been processed." %(idx+1, len(quatrains)))
        print ("Hold on. This may take some time ...")
        self.kw_model = models.Word2Vec(seg_lists, size = 512, min_count = 20)
        self.kw_model.save(kw_model_path)
        print("model saved")

    def old_expand(self, words, num, positive):
        #positive = list(filter(lambda w: w in self.kw_model.wv, words))
        similars = self.kw_model.wv.most_similar(positive = positive) \
                if len(positive) > 0 else []
        print(similars)
        words.extend(pair[0] for pair in similars[:min(len(similars), num-len(words))])
        print(words)
        if len(words) < num:
            _prob_sum = sum(1./(i+1) for i in range(len(self.ranks)))
            _prob_sum -= sum(1./(self.ranks[word]+1) for word in words)
            while len(words) < num:
                prob_sum = _prob_sum
                for word, rank in self.ranks.items():
                    if word in words:
                        continue
                    elif prob_sum * random() < 1./(rank+1):
                        words.append(word)
                        break
                    else:
                        prob_sum -= 1./(rank+1)
        shuffle(words)
        return words

    def expand(self, words, num):
        positive = []
        for word in words:
            for single in word:
                positive.append(single)
        similars = self.model.wv.most_similar(positive = positive)
        words.extend(pair[0] for pair in similars[:min(len(similars), num-len(words))])
        if len(words) < num:
            _prob_sum = sum(1./(i+1) for i in range(len(self.rank)))
            _prob_sum -= sum(1./(self.rank[word]+1) for word in words)
            while len(words) < num:
                prob_sum = _prob_sum
                for word, _rank in self.rank.items():
                    if word in words:
                        continue
                    elif prob_sum * random() < 1./(_rank+1):
                        words.append(word)
                        break
                    else:
                        prob_sum -= 1./(_rank+1)
        shuffle(words)
        return words

    def plan(self, text):
        def check(words):#检查词中的字是否在字典中
            for word in words:
                if word not in self.rank:
                    return False
            return True
        def extract(sentence):
            return list(filter(lambda x: check(x), jieba.lcut(sentence)))
        def _rank(words):#
            r = 1
            length = 0.00001
            for word in words:
                if word not in self.stop_words:
                    r = r + self.rank[word]
                    length = length+1
            r = r/length
            if len(words) >= 2:
                r -= 5000
            return r
        print(jieba.lcut(text))
        _keywords = extract(text)#词级别关键词
        print(_keywords)
        keywords = []
        for w1 in _keywords:#去掉重复关键词
            flag = True
            for w2 in _keywords:
                if len(w1) < len(w2) and w1 in w2:
                    flag = False
                    break
            if flag == True:
                keywords.append(w1)
        print(keywords)
        for word in text:#字级别关键词
            flag = True
            if word not in self.ranks:
                flag = False
            else:
                for words in keywords:
                    if word in words:
                        flag = False
                        break
            if flag == True:
                keywords.append(word)
        #print(keywords)
        keywords = sorted(keywords, key = lambda x: _rank(x))
        print(keywords)
        words = [keywords[idx] for idx in \
                filter(lambda i: 0 == i or keywords[i] != keywords[i-1], range(len(keywords)))]
        if len(words) == 0:#输入没有字典中的字
            word = self.int2ch(random.randint(1,4000))
            words.append(word)
        print(words)
        if len(words) < 4:
            positive = []
            for word in words:
                if word in self.kw_model.wv:
                    positive.append(word)
                else:
                    for w in word:
                        if w in self.kw_model.wv:
                            positive.append(w)
            print(positive)
            words = self.old_expand(words, 4, positive)
        else:
            while len(words) > 4:
                words.pop()
        return words

if __name__ == '__main__':
    planner = Planner()
    while(True):
        s = input()
        print(planner.plan(s))