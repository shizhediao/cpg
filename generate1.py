#! /usr/bin/env python
#-*- coding:utf-8 -*-

from utils import *
from vocab import *
from word2vec import get_word_embedding
from data_utils import *
from collections import deque
from model import *
import torch
import torch.nn as nn
import random
from pypinyin import pinyin, Style
from math import floor

hidden_size = 128
_model_path = os.path.join(save_dir, 'seq2seq_8_31_2_40.pt')
_refine_path = os.path.join(save_dir, 'refine_8_31_2_40.pt')

rhythm_1 = [[2,0,2,1,0,0,1],[2,1,0,0,1,1,0],[2,1,2,0,0,1,1],[2,0,2,1,1,0,0]]
rhythm_2 = [[2,0,2,1,1,0,0],[2,1,0,0,1,1,0],[2,1,2,0,0,1,1],[2,0,2,1,1,0,0]]
rhythm_3 = [[2,1,2,0,0,1,1],[2,0,2,1,1,0,0],[2,0,2,1,0,0,1],[2,1,0,0,1,1,0]]
rhythm_4 = [[2,1,0,0,1,1,0],[2,0,2,1,1,0,0],[2,0,2,1,0,0,1],[2,1,0,0,1,1,0]]
rhythm = [rhythm_1, rhythm_2, rhythm_3, rhythm_4]

class Generator:

    def __init__(self):
        self.encoder = Encoder(VOCAB_SIZE, 128, hidden_size, n_layers=2, dropout=0.5).cuda()
        self.decoder = Decoder(128, hidden_size, VOCAB_SIZE, n_layers=2, dropout=0.5).cuda()
        self.seq2seq = Seq2Seq(self.encoder, self.decoder).cuda()
        self.refine = Seq2Seq(self.encoder, self.decoder).cuda()
        self.seq2seq.load_state_dict(torch.load(_model_path))
        self.refine.load_state_dict(torch.load(_refine_path))
        self.ya = 0
        self.yalist = []
        self.word_list1, self.word_list2 = find_repeat_word()

    def find_best_word(self, sentence, output, sentence_id, word_id, r_id):
        rhy = rhythm[r_id][sentence_id][word_id]
        int2ch, ch2int = get_vocab()
        output = output.data
        while(True):
            idx = output.max(1)[1][0]
            word = int2ch[idx]
            cnt = 0
            for w in sentence:
                if word == w:
                    cnt = cnt + 1
            if cnt >= 2:
                output[0][idx] = -100000
                continue
            #if word not in self.word_list2:
            flag = True
            for w in sentence:
                if word == w:
                    output[0][idx] = -100000
                    flag = False
                    break
            if flag == False:
                continue
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
        br = []
        int2ch, ch2int = get_vocab()
        #print(keywords)
        rhythm_id = floor(random.random()*4)
        for idx, keyword in enumerate(keywords):
            sentence = ''
            output = Variable(torch.LongTensor([0])).cuda()
            #print(output)
            kw = Variable(torch.LongTensor([[ch2int[ch]] for ch in keyword])).cuda()
            #print(kw)
            encoder_output, hidden = self.seq2seq.encoder(kw)
            hidden = hidden[:2]
            for t in range(7):
                #print(t)
                output, hidden, attn_weights = self.seq2seq.decoder(output, hidden, encoder_output)
                #print(output.data)
                #print(output.data.max(1)[1])
                #print(output.data.max(1)[1])
                sentence += int2ch[output.data.max(1)[1][0]]
                #print(sentence)
                output = Variable(output.data.max(1)[1])
            br.append(sentence)
            sentence = Variable(torch.LongTensor([[ch2int[ch]] for ch in sentence])).cuda()
            encoder_output, hidden = self.refine.encoder(sentence)
            hidden = hidden[:2]
            sentence = ''
            for t in range(7):
                output, hidden, attn_weights = self.refine.decoder(output, hidden, encoder_output)
                word = self.find_best_word(sentence, output, idx, t, rhythm_id)
                #sentence += int2ch[output.data.max(1)[1][0]]
                sentence += word
                #output = Variable(output.data.max(1)[1])
                output = Variable(torch.LongTensor([ch2int[word]])).cuda()
            sentences.append(sentence)
            
        return br, sentences


if __name__ == '__main__':
    generator = Generator()
    train_data = get_train_data()
    test_data = get_test_data()
    #for row in kw_train_data[:100]:
    #    print(row)
    #    print(generator.generate(row))
    #    print()
    print(len(train_data))
    i = 0
    kw = []
    s = []
    with codecs.open('out_r_8_31_2.txt', 'w', 'utf-8') as fout:
        for row in train_data:
            i = i+1 
            if len(row['sentence']) != 7:
                continue
            s.append(row['sentence'])
            kw.append(row['keyword'])
            if i%100 == 0:
                print(i)
            if i%4 == 0:
                generator.ya = 0
                generator.yalist = []
                fout.write(' '.join(kw))
                fout.write('\n')
                fout.write(' '.join(s))
                fout.write('\n')
                br, ar = generator.generate(kw)
                fout.write(' '.join(br))
                fout.write('\n')
                fout.write(' '.join(ar))
                fout.write('\n')
                kw = []
                s = []
