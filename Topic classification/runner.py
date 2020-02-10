from collections import defaultdict
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
import numpy as np
import sys
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import time
from preprocessing import tokenizer, build_vocab, MyDataSet, collate_sequence
from model import CNN

def train_epoch(cnn, train_loader, criterion, optimizer):
    
    cnn.train()
    cnn.to(device)

    train_loss = 0.0

    for batch_idx, (train_X, train_y) in enumerate(train_loader):
        optimizer.zero_grad()
        train_X = train_X.to(device)
        train_y = train_y.to(device)
        output = cnn(train_X)
        loss = criterion(output, train_y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)
        correct_prediction = (predicted == train_y).sum().item()


        if batch_idx % 1000 == 0:
            print("accuracy is:", correct_prediction / len(train_X))
            print('Batch: {} [{}/{} ({:}%, \tLoss: {:.6f}'.format(
                batch_idx, batch_idx * len(train_X), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader.dataset), loss.item()))
    train_loss /= len(train_loader)
    print('Training Loss: ', train_loss)
    return train_loss


def val_epoch(cnn, val_loader, criterion):
    with torch.no_grad():
        cnn.eval()
        cnn.to(device)

        cnn.train()
        cnn.to(device)

        val_loss = 0.0
        total_prediction = 0.0
        correct_prediction = 0.0

        for batch_idx, (val_X, val_y) in enumerate(val_loader):
            val_X = val_X.to(device)
            val_y = val_y.to(device)
            output = cnn(val_X)
            _, predicted = torch.max(output.data, 1)
            total_prediction += val_y.size(0)
            correct_prediction += (predicted == val_y).sum().item()

            loss = criterion(output, val_y).detach()
            val_loss += loss.item()

            if batch_idx % 100 == 0:
                print('Batch: {} [{}/{} ({:}%, \tLoss: {:.6f}'.format(
                    batch_idx, batch_idx * len(val_X), len(val_loader.dataset),
                               100. * batch_idx / len(val_loader.dataset), loss.item()))
        val_loss /= len(val_loader)
        acc = (correct_prediction / total_prediction) * 100.0
        print('Validation Loss: ', val_loss)
        print('Validation accuracy', acc)
        return val_loss, acc


def test_epoch(cnn, test_loader):

    prediction = []
    test_X, _ = test_loader
    for i in range(len(test_X)):

        test_x = test_x.to(device)
        output = cnn(test_x)
        _, predict = torch.max(output.data, 1)

        prediction.append(predict.item())
    return prediction

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    train_X, train_y, val_X, val_y, X_test, embedding_matrix = build_vocab("topicclass/")

    train_dataset = MyDataSet(train_X, train_y)
    val_dataset = MyDataSet(val_X, val_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64, collate_fn = collate_sequence)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=64, collate_fn = collate_sequence)
    vocab_size = len(embedding_matrix)
    learning_rate = 1e-5
    num_class = 17
    embed_len = 200
    in_channels = 1
    out_channels = 128
    stride = 1
    padding = 0
    keep_prob = 0.5
    kernel_size = [3, 6, 9]
    #kernel_size = [3, 4, 5]
    model = CNN(num_class, out_channels, kernel_size, stride, padding, keep_prob, vocab_size, embed_len, embedding_matrix)
    loss_fn = F.cross_entropy
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    Train_loss = []
    Val_acc = []
    Val_loss = []

    for i in range(30):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = val_epoch(model, val_loader, criterion)
        Train_loss.append(train_loss)
        Val_loss.append(val_loss)
        Val_acc.append(val_acc)
        torch.save(model.state_dict(), "experiment1{:03d}epoch_{}.w".format(i, int(val_acc * 100)))