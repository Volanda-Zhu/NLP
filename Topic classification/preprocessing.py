import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from collections import defaultdict
from collections import Counter
import random
from torch.utils import data
import config
import time
from torch.nn.utils.rnn import *
from torch.utils.data import Dataset, DataLoader
import os
import spacy


topics = ['sports and recreation', 'social sciences and society', 'media and drama', 'warfare', \
          'engineering and technology', 'language and literature', 'history', 'mathematics', \
          'philosophy and religion', 'art and architecture', 'video games', 'miscellaneous', \
          'music', 'natural sciences', 'agriculture, food and drink', 'geography and places']


spacy_en = spacy.load('en_core_web_sm')
def tokenizer(text): 
    return [tok.text for tok in spacy_en.tokenizer(text)]


def build_vocab(directory):
    w2i = {}
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    i = 0
    file_lst = os.listdir(directory)
    for filename in file_lst:
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename)) as f:
                for line in f:
                    topic, context = line.lower().strip().split(' ||| ')
                    if topic == 'media and darama':
                        topic = 'media and drama'
                    words = tokenizer(context)
                    if filename.endswith("_train.txt"):
                        y_train.append(topics.index(topic))
                    elif filename.endswith("_valid.txt"):
                        y_val.append(topics.index(topic))
                    for word in words:
                        if word not in w2i.keys():
                            w2i[word] = i
                            i += 1
                    if filename.endswith("_train.txt"):
                        X_train.append([w2i[word] if word in w2i.keys() else w2i["UNK"] for word in words])
                    elif filename.endswith("_valid.txt"):
                        X_val.append([w2i[word] if word in w2i.keys() else w2i["UNK"] for word in words])
                    else:
                        X_test.append([w2i[word] if word in w2i.keys() else w2i["UNK"] for word in words])
    print("There are ", len(w2i), " words in the dictionary")
    glove_embedding = {}
    with open("./glove.6B.200d.txt") as f:

        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove_embedding[word] = coefs

    embedding_matrix = np.zeros((len(w2i), config.EMB_DIM))

    for i, word in enumerate(w2i.keys()):
        if word in glove_embedding.keys():
            embedding_matrix[i] = glove_embedding[word]
        else:
            embedding_matrix[i] = random.choice(list(glove_embedding.values()))
    return X_train, y_train, X_val,y_val, X_test, embedding_matrix

class MyDataSet(Dataset):
    def __init__(self,X, Y):
        self.X = X
        self.Y = Y
    def __getitem__(self,i):
        return self.X[i], self.Y[i]
    def __len__(self):
        return len(self.X)

def collate_sequence(seq_list):
    inputs,targets = zip(*seq_list)

    inputs = [torch.LongTensor(input) for input in inputs]
    targets = torch.LongTensor(targets)

    inputs = pad_sequence(inputs, batch_first = True)

    return inputs,targets

if __name__ == "__main__":
    X_train, y_train, X_val,y_val, X_test, embedding_matrix = build_vocab("topicclass/")
    
