from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import LSTM, Dense, Activation
from keras.preprocessing.sequence import pad_sequences
import argparse
from pickle import dump
from pickle import load
import itertools
import numpy as np
import random
import sys
import os
import re

def load_sonnets(datafile="data/shakespeare.txt"):
    sonnets = []
    with open(datafile) as text:
        start=1
        lines = [line.strip(' ').lower() for line in text]
        for i, line in enumerate(lines):
            if line[:-1].isdigit():
                start = i + 1
            elif not line[:-1]:
                end = i
                sonnets.append(lines[start:end])
            elif i == len(lines)-1:
                end = i+1
                sonnets.append(lines[start:end])
    return sonnets

def get_sequences(sonnets, length=40, step=1):
    sequences = []
    # split each sonnet into sequences of length+1 = 41 and merge. 
    # do each sonnet separately because of independence concerns
    for son in sonnets:
        # clean each sonnet and get rid of new lines to combine into one big list
        text = ' '.join(son)
        tokens = text.split()
        text = ' '.join(tokens)
        text = re.sub(r'[^\w\'\s\,]', '', text)
        for i in range(length, len(text), step):
            seq = text[i-length:i+1]
            sequences.append(seq)
    return sequences

def training_data(sequences, length=40):
    # all the possible characters, sorted and indexed
    chars = sorted(set([c for s in sequences for c in s]))
    vocabulary = dict((c, i) for i, c in enumerate(chars))
    dump(vocabulary, open('vocabulary.pkl', 'wb'))
    # input will be a list with each character in each sequence mapped to a 1 of k encoding of the vocabulary
    # input will be list of list of one hot vectors
    # output will be a list of one hot vectors corresponding to the next character following each sequence
    # x = np.zeros((len(sequences), length, len(vocabulary)))
    # y = np.zeros((len(sequences), len(vocabulary)))
    encoded = []
    for i, seq in enumerate(sequences):
        encoded.append([vocabulary[char] for char in seq])
    encoded = np.array(encoded)
    # x is the first 40 characters and y is the next character, the 41st
    inp, out = encoded[:,:-1], encoded[:,-1]
    X = np.array([to_categorical(i, num_classes=len(vocabulary)) for i in inp])
    y = to_categorical(out, num_classes=len(vocabulary))

    return X,y

def train_model(X, y, epochs=100, layers=1):
    model = Sequential()
    model.add(LSTM(150, input_shape=(X.shape[1], X.shape[2])))
    for i in range(1, layers):
        model.add(LSTM(return_sequences=layers>1))
    model.add(Dense(X.shape[2], activation='softmax'))
    print(model.summary())
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(X, y, epochs=epochs, verbose=2)
    model.save('model.h5')


def generate_sonnet(model, mapping, seq_length=40, seed = "shall i compare thee to a summer's day?", n_chars=700):
    sonnets = []
    for diversity in [0.25, 0.75, 1.5]:
        print('temperature:', diversity)
        generated = seed
        for i in range(n_chars):
            # encode the characters as integers
            encoded = [mapping[char] for char in seed]
            # truncate sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
            # one hot encode
            encoded = to_categorical(encoded, num_classes=len(mapping))
            # predict character
            yhat = model.predict_classes(encoded, verbose=0)
            # reverse map integer to character
            out_char = ''
            for char, index in mapping.items():
                if index == yhat:
                    out_char = char
                    break
            # add to generated sonnet
            generated += char
        print(generated)
        sonnets.append(generated)
    return sonnets



if __name__ == '__main__':
    # sonnets = load_sonnets()
    # sequences = get_sequences(sonnets)
    # X, y = training_data(sequences)
    # train_model(X,y)

    # # load the model
    # model = load_model('model.h5')
    # # load the mapping
    mapping = load(open('mapping.pkl', 'rb'))
    print(mapping)
    # generate_sonnet(model, mapping)
