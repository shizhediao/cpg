#! /usr/bin/env python
# -*- coding:utf-8 -*-
#最新的生成器

from utils import *
from vocab import *
from word2vec import get_word_embedding
from data_utils import *
from collections import deque
from gensim import models
from model import *
import torch
import codecs
from pypinyin import pinyin, Style
import torch.nn as nn
import numpy as np
from math import floor
from gensim import models
from discriminator import Discriminator

hidden_size = 256
_model_path = os.path.join(save_dir, 'seq2seq_10_11_20.pt')
w2v_dir = os.path.join(data_dir, 'word2vec.model')
kw_model_path = os.path.join(data_dir, 'kw_model.bin')

rhythm_1 = [[2, 0, 2, 1, 0, 0, 1], [2, 1, 0, 0, 1, 1, 0], [2, 1, 2, 0, 0, 1, 1], [2, 0, 2, 1, 1, 0, 0]]
rhythm_2 = [[2, 0, 2, 1, 1, 0, 0], [2, 1, 0, 0, 1, 1, 0], [2, 1, 2, 0, 0, 1, 1], [2, 0, 2, 1, 1, 0, 0]]
rhythm_3 = [[2, 1, 2, 0, 0, 1, 1], [2, 0, 2, 1, 1, 0, 0], [2, 0, 2, 1, 0, 0, 1], [2, 1, 0, 0, 1, 1, 0]]
rhythm_4 = [[2, 1, 0, 0, 1, 1, 0], [2, 0, 2, 1, 1, 0, 0], [2, 0, 2, 1, 0, 0, 1], [2, 1, 0, 0, 1, 1, 0]]
rhythm = [rhythm_1, rhythm_2, rhythm_3, rhythm_4]

if torch.cuda.is_available():
    CUDA = True
else:
    CUDA = False
class Generator:

    def __init__(self):
        if CUDA:
            self.encoder = Encoder(VOCAB_SIZE, 300, hidden_size, n_layers=1, dropout=0.5).cuda()
            self.decoder = Decoder(300, hidden_size, VOCAB_SIZE, n_layers=1, dropout=0.5).cuda()
            self.seq2seq = Seq2Seq(self.encoder, self.decoder).cuda()
        else:
            self.encoder = Encoder(VOCAB_SIZE, 300, hidden_size, n_layers=1, dropout=0.5)
            self.decoder = Decoder(300, hidden_size, VOCAB_SIZE, n_layers=1, dropout=0.5)
            self.seq2seq = Seq2Seq(self.encoder, self.decoder)
        self.seq2seq.load_state_dict(torch.load(_model_path, map_location='cpu'))
        self.ya = 0
        self.yalist = []
        self.word_model = models.Word2Vec.load(kw_model_path)
        # self.word_list1, self.word_list2 = find_repeat_word()
        # self.model = models.Word2Vec.load(w2v_dir)
        # self.discriminator = Discriminator(embedding_dim=256, hidden_dim=256, vocab_size=VOCAB_SIZE, max_seq_len=7,gpu=True).cuda()
        # self.discriminator.load_state_dict(torch.load('./save/dis_9_14_b32_10.pt'))
        # with codecs.open('w_rank.json', 'r', 'utf-8') as fin:
        #    self.w_rank = json.load(fin)

    def find_best_word(self, sentence, output, sentence_id, word_id, r_id, teachword):
        teach_len = len(teachword)
        if word_id < teach_len:
            return teachword[word_id]
        rhy = rhythm[r_id][sentence_id][word_id]
        int2ch, ch2int = get_vocab()
        output = output.data
        # print(output.max(1))
        # RANK = 100
        # ORANK = 100
        t = 0
        while (True):
            if t == 0:
                first = output.max(1)[1][0]
            t = t + 1
            if t == 100:
                return int2ch[first]
            idx = output.max(1)[1][0]
            word = int2ch[idx]
            flag = False
            # if word in keywords:
            sent = [ch2int[w] for w in sentence]
            sent.extend([idx])
            if CUDA:
                sent = Variable(torch.LongTensor([sent])).cuda()
            else:
                sent = Variable(torch.LongTensor([sent]))
            # p = self.discriminator(sent).data[0][0]
            # if p < 0.7:
            #    output[0][idx] = -1000000
            #    continue
            flag = True
            # if word_id > 0:
            #   try:
            #      if self.model.similarity(word,sentence[word_id-1]) < -0.1:
            #         output[0][idx] = -1000000
            #        continue
            # except KeyError:
            #   output[0][idx] = -1000000
            #  continue
            if flag == False:
                if output[0][idx] <= -ORANK:
                    RANK = RANK + 100
                    ORANK = RANK
                if word not in self.w_rank or self.w_rank[word] > RANK:
                    output[0][idx] = -RANK
                    RANK = RANK + 1
                    continue
            cnt = 0
            for w in sentence:
                if word == w:
                    cnt = cnt + 1
            if cnt >= 2:
                output[0][idx] = -1000000
                continue
            # if word not in self.word_list2:
            flag = True
            for w in sentence:
                if w == word:
                    output[0][idx] = -1000000
                    flag = False
            if flag == False:
                continue
            # if word not in self.word_list2:
            #    if sentence:
            #        if word == sentence[-1]:
            #            output[0][idx] = -10000
            #            continue
            py = pinyin(word, style=Style.TONE3)[-1][-1][-1]
            if rhy == 2:
                break
            elif rhy == 1:
                if py == '3' or py == '4':
                    break
            elif rhy == 0:
                if py == '1' or py == '2':
                    if word_id == 6:
                        if self.ya == 0:
                            self.ya = pinyin(word, style=Style.FINALS)
                            self.yalist.append(word)
                            break
                        elif self.ya == pinyin(word, style=Style.FINALS) and word not in self.yalist:
                            self.yalist.append(word)
                            break
                    else:
                        break
            output[0][idx] = -10000
        return word

    def generate(self, keywords):
        sentences = []
        ss = []
        int2ch, ch2int = get_vocab()
        # print(keywords)
        rhythm_id = floor(random.random() * 4)
        for idx, keyword in enumerate(keywords):
            if keyword in self.word_model.wv:
                teachword = self.word_model.most_similar(keyword)
                for word in teachword:
                    if word[0] not in keywords:
                        flag = 1
                        for w in word[0]:
                            if w not in ch2int:
                                flag = 0
                                break
                        if flag == 1:
                            teachword = word[0]
                            break
            else:
                flag1 = False
                for w in keyword:
                    if w in self.word_model.wv:
                        teachword = self.word_model.most_similar(w)
                        for word in teachword:
                            if word[0] not in keywords:
                                flag2 = 1
                                for w2 in word[0]:
                                    if w2 not in ch2int:
                                        flag2 = 0
                                        break
                                if flag2 == 1:
                                    flag1 = True
                                    teachword = word[0]
                                    break
                    if flag1 == True:
                        break
                flag1 = False
                if flag1 == False:
                    teachword = keyword
            print(teachword)
            sentence = u''
            if CUDA:
                output = Variable(torch.LongTensor([0])).cuda()
                # print(output)
                kw = Variable(torch.LongTensor([[ch2int[ch]] for ch in keyword])).cuda()
            else:
                output = Variable(torch.LongTensor([0]))
                # print(output)
                kw = Variable(torch.LongTensor([[ch2int[ch]] for ch in keyword]))
            # print(kw)
            encoder_output, hidden = self.seq2seq.encoder(kw)
            # print(hidden)
            hidden = hidden[:2]
            print(1)
            for t in range(7):
                # print(t)
                output, hidden, attn_weights = self.seq2seq.decoder(output, hidden, encoder_output)
                # print(output.data)
                # print(output.data.max(1)[1])
                # print(output.data.max(1)[1])
                word = self.find_best_word(ss, output, idx, t, rhythm_id, teachword)
                # sentence += int2ch[output.data.max(1)[1][0]]
                sentence += word
                ss += word
                # output = Variable(output.data.max(1)[1])
                if CUDA:
                    output = Variable(torch.LongTensor([ch2int[word]])).cuda()
                else:
                    output = Variable(torch.LongTensor([ch2int[word]]))
                # sentence += int2ch[output.data.max(1)[1][0]]
                # print(sentence)
                # output = Variable(output.data.max(1)[1])
            sentences.append(sentence)

        return sentences


if __name__ == '__main__':
    generator = Generator()
    train_data = get_train_data()
    kw_train_data = get_kw_train_data()
    word2vec = models.Word2Vec.load('word2vec.model')
    # for row in kw_train_data[:100]:
    #    print(row)
    #    print(generator.generate(row))
    #    print()
    print(len(train_data))
    i = 0
    kw = []
    s = []
    os = []
    ms = []
    data = []
    with codecs.open('dis_train.txt', 'w', 'utf-8') as fout:
        for row in train_data:
            i = i + 1
            if len(row['sentence']) != 7:
                continue
            s.append(row['sentence'])
            kw.append(row['keyword'])
            if i % 100 == 0:
                print(i)
            if i % 4 == 0:
                generator.yalist = []
                generator.ya = 0
                # fout.write(' '.join(kw))
                # fout.write('\n')
                fout.write('\n'.join(s))
                fout.write('\n')
                sentences = generator.generate(kw)
                fout.write('\n'.join(sentences))
                fout.write('\n')
                kw = []
                s = []
