import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import codecs
import pdb
import numpy as np
from vocab import get_vocab, VOCAB_SIZE
import random

class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.5):
        super(Discriminator, self).__init__()
        self.hidden_dim = embedding_dim
        self.max_seq_len_ = max_seq_len
        self.gpu = gpu
        self.L = 1
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=self.L, bidirectional=False, dropout=dropout)
        self.gru2hidden = nn.Linear(self.L*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(self.L*2*1, batch_size, self.hidden_dim))
        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, input, hidden=None):
        #print(input)
        #if hidden == None:
        #    hidden = self.init_hidden(128)
        emb = self.embeddings(input)
        emb = emb.permute(1, 0, 2)
        output, hidden = self.gru(emb, hidden)
        hidden = hidden.permute(1, 0, 2).contiguous()
        out = self.gru2hidden(hidden.view(-1, self.L*self.hidden_dim))
        out = F.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)
        out = F.sigmoid(out)
        return out


class TrainDataset(Dataset):
    def __init__(self):
        self.data_s = np.load('dis_t_sentence.npy')
        self.data_l = np.load('dis_t_label.npy')

    def __getitem__(self, index):
        return self.data_s[index], self.data_l[index]

    def __len__(self):
        return len(self.data_s)

class TestDataset(Dataset):
    def __init__(self):
        self.data_s = np.load('dis_v_sentence.npy')
        self.data_l = np.load('dis_v_label.npy')

    def __getitem__(self, index):
        return self.data_s[index], self.data_l[index]

    def __len__(self):
        return len(self.data_s)

def gen_dataset():
    t_s = []
    t_l = []
    v_s = []
    v_l = []
    line = []
    int2ch, ch2int = get_vocab()
    i = 0
    value = 1
    with codecs.open("dis_train.txt", "r", "utf-8") as fin:
        while(True):
            i = i + 1
            if i%1000 == 0:
                print(i)
            line = fin.readline().split()
            #print(line)
            if not line:
                break
            line = line[0]
            sentence = [ch2int[word] for word in line]
            #print(sentence)
            ss = []
            ll = []
            #for i in range(3,7):
            ss.append(sentence)
            ll.append(value)
            if i%4 == 0:
                if value == 0:
                    value = 1
                else:
                    value = 0
            if random.random() < 0.8:
                t_s.extend(ss)
                t_l.extend(ll)
            else:
                v_s.extend(ss)
                v_l.extend(ll)
    np.save("dis_t_sentence.npy", t_s)
    np.save("dis_t_label.npy", t_l)
    np.save("dis_v_sentence.npy", v_s)
    np.save("dis_v_label.npy", v_l)

def train(model, optimizer, train_loader, test_loader):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model.train()
    total_loss = 0
    b = 0
    acc = 0
    zo = 0
    tot = 0
    answer = []
    ori = []
    for sentence, label in train_loader:
        b = b + 1
        optimizer.zero_grad()
        #sentence = sentence.t()
        #label = label.t()
        sentence = Variable(sentence.type(torch.LongTensor)).cuda()
        label = Variable(label.type(torch.FloatTensor)).cuda()
        output = model(sentence)
        #print(label.data[0])
        for i in range(len(output)):
            tot = tot + 1
            #if label.data[i] == 1:
            #    zo = zo + 1
            ori.extend([label.data[i]])
            if output.data[i][0] > 0.5:
                answer.extend([1])
                if label.data[i] == 1:
                    acc = acc + 1
            else:
                answer.extend([0])
                if label.data[i] == 0:
                    acc = acc + 1
        loss_fn = nn.BCELoss()
        loss = loss_fn(output, label)
        loss.backward()
        total_loss += loss.data[0]
        clip_grad_norm(model.parameters(), 10)
        optimizer.step()
        if b % 200 == 0:
            total_loss = total_loss/200
            print(b, total_loss)
            total_loss = 0
    #print(answer[:100])
    #print(ori[:100])
    #print(acc, tot)
    print('accuracy: %f' % (acc/tot))
    #print(zo, tot-zo)
    b = 0
    total_loss = 0
    acc = 0
    acc1 = 0
    acc2 = 0
    tot = 0
    answer = []
    ori = []
    prob = []
    for sentence, label in test_loader:
        b = b + 1
        optimizer.zero_grad()
        #sentence = sentence.t()
        #label = label.t()
        sentence = Variable(sentence.type(torch.LongTensor)).cuda()
        label = Variable(label.type(torch.FloatTensor)).cuda()
        output = model(sentence)
        for i in range(len(output)):
            tot = tot + 1
            if label.data[i] == 1:
                zo = zo+1
            ori.extend([label.data[i]])
            prob.extend([output.data[i][0]])
            if output.data[i][0] > 0.5:
                answer.extend([1])
                if label.data[i] == 1:
                    acc1 = acc1 + 1
            else:
                answer.extend([0])
                if label.data[i] == 0:
                    acc2 = acc2 + 1
        loss_fn = nn.BCELoss()
        loss = loss_fn(output, label)
        total_loss += loss.data[0]
        if b % 200 == 0:
            total_loss = total_loss/200
            print(b, total_loss)
            total_loss = 0
    #print(answer[:100])
    #print(ori[:100])
    #print(acc, tot)
    #print(prob[:100])
    print(acc1, zo, acc2, tot-zo)
    print('test acc: %f' % ((acc1+acc2)/tot))

def main():
    epoches = 100
    train_dataset = TrainDataset()
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_dataset = TestDataset()
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)
    discriminator = Discriminator(embedding_dim=256, hidden_dim=256, vocab_size=VOCAB_SIZE, max_seq_len=7,gpu=True).cuda()
    optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    #discriminator.load_state_dict(torch.load('./save/dis_9_14_b32_10.pt'))
    print(discriminator)
    for i in range(1, 1+epoches):
        print(i)
        train(discriminator, optimizer, train_loader, test_loader)
        if i%10 == 0:
            torch.save(discriminator.state_dict(), './save/dis_9_28_b32_%d.pt' % (i))

if __name__ == "__main__":
    #gen_dataset()
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped')
