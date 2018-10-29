#训练
import os
import math
import argparse
import torch
from torch.utils.data import *
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from model import *
from data_utils import *

hidden_size = 256


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.00001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='initial learning rate')
    return p.parse_args()


class CustomDataset(Dataset):
    def __init__(self):
        self.data_s = np.load('sentence7t.npy')
        self.data_kw = np.load('keyword7t.npy')

    def __getitem__(self, index):
        return self.data_s[index], self.data_kw[index]

    def __len__(self):
        return len(self.data_s)

class TestDataset(Dataset):
    def __init__(self):
        self.data_s = np.load('sentence7v.npy')
        self.data_kw = np.load('keyword7v.npy')

    def __getitem__(self, index):
        return self.data_s[index], self.data_kw[index]

    def __len__(self):
        return len(self.data_s)

def train(model, optimizer, train_loader, grad_clip, test_loader):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model.train()
    total_loss = 0
    b = 0
    acc = 0
    total = 0
    for sentence, keyword in train_loader:
        #print(b)
        b = b+1
        optimizer.zero_grad()
        keyword = keyword.t()#.resize_(3,keyword.size(0),1)
        sentence = sentence.t()#.resize_(8,sentence.size(0),1)
        keyword = Variable(keyword.type(torch.LongTensor)).cuda()
        sentence = Variable(sentence.type(torch.LongTensor)).cuda()
        output = model(keyword, sentence, 1)
        #print('a')
        out_idx = output.max(2)[1]
        #print(out_idx[1:][0][0])
        #print(sentence[1:][0][127])
        ll = out_idx.size(1)
        for i in range(7):
            for j in range(ll):
                #print(out_idx.data[1:][i][j])
                if out_idx.data[1:][i][j] == sentence.data[1:][i][j]:
                    acc = acc + 1
                #print(sentence[1:][i][j][0])
                total = total + 1
        #print(output[1:].view(-1,VOCAB_SIZE))
        loss = F.cross_entropy(output[1:].view(-1, VOCAB_SIZE),
                               sentence[1:].contiguous().view(-1))
        #print('b')
        loss.backward()
        #print('c')
        total_loss += loss.data[0]
        #print('d')
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        if b % 20 == 0:
            total_loss = total_loss/20
            print(b, total_loss)
            total_loss = 0
    print('train accuracy:', acc/total)
    acc = 0
    total = 0
    #for sentence, keyword in train_loader:
    #    keyword = keyword.t()#.resize_(3,keyword.size(0),1)
    #    sentence = sentence.t()#.resize_(8,sentence.size(0),1)
     #   keyword = Variable(keyword.type(torch.LongTensor)).cuda()
      #  sentence = Variable(sentence.type(torch.LongTensor)).cuda()
       # output = model(keyword, sentence, 0)
    #    out_idx = output.max(2)[1]
     #   ll = out_idx.size(1)
      #  for i in range(7):
       #     for j in range(ll):
        #       #print(out_idx.data[1:][i][j])
         #       if out_idx.data[1:][i][j] == sentence.data[1:][i][j]:
          #          acc = acc + 1
                #print(sentence[1:][i][j][0])
           #     total = total + 1
    #print('test_accuracy', acc/total)
        

def main():
    args = parse_arguments()
    dataset = CustomDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=False)
    test_dataset = TestDataset()
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    encoder = Encoder(VOCAB_SIZE, 300, hidden_size, n_layers=1, dropout=0.5).cuda()
    decoder = Decoder(300, hidden_size, VOCAB_SIZE, n_layers=1, dropout=0.5).cuda()
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    #seq2seq.load_state_dict(torch.load('./save/seq2seq_9_7_1_40.pt'))
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)

    print(seq2seq)

    for i in range(1, args.epochs+1):
        print(i)
        train(seq2seq, optimizer, train_loader, args.grad_clip, test_loader)
        if i % 10 == 0:
            torch.save(seq2seq.state_dict(), './save/seq2seq_10_8_%d.pt' % (i))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped')
