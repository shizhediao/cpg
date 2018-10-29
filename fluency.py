#暂时不用
from utils import *
from vocab import get_vocab, VOCAB_SIZE
from quatrains import get_quatrains
from gensim import models
import numpy as np

if __name__ == '__main__':
    poems = get_quatrains()
    int2ch, ch2int = get_vocab()
    model = models.Word2Vec.load('word2vec.model')
    #print(poems)
    simi = []
    for poem in poems:
        sentences = poem['sentences']
        for sentence in sentences:
            #print(sentence)
            for word1 in sentence:
                for word2 in sentence:
                    try:
                        simi.append(model.similarity(word1, word2))
                    except KeyError:
                       continue
    print(np.array(simi))
    np.save(simi, 'simi.npy')
