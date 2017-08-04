#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Inspired by http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import argparse
from operator import itemgetter

seq_length = 100
train_filename = "congratulations.txt" # http://lib.ru/ANEKDOTY/pozdrawleniya.txt

def create_model(seq_length, n_vocab, weights=''):
    # define the LSTM model
    # model = Sequential()
    # model.add(LSTM(256, input_shape=(seq_length, 1)))
    # model.add(Dropout(0.2))
    # model.add(Dense(n_vocab, activation='softmax'))
    model = Sequential()
    model.add(LSTM(256, input_shape=(seq_length, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(n_vocab, activation='softmax'))
    # load the network weights
    if weights != '':
        model.load_weights(weights)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def load_train_data(filename):
    # load ascii text and covert to lowercase
    raw_text = open(filename).read()
    raw_text = raw_text.lower()
    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)
    # prepare the dataset of input to output pairs encoded as integers
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)
    return {'X': X, 'y': y, 'n_vocab': n_vocab, 'dataX': dataX,
            'char_to_int': char_to_int, 'int_to_char': int_to_char}


def train():
    X, y, n_vocab, dataX = itemgetter('X', 'y', 'n_vocab', 'dataX')(
        load_train_data(train_filename))

    # define the LSTM model
    model = create_model(seq_length, n_vocab)
    # define the checkpoint
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')

    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)


def generate_text(weights_file):
    X, y, n_vocab, dataX, int_to_char = itemgetter(
        'X', 'y', 'n_vocab', 'dataX', 'int_to_char')(
            load_train_data(train_filename))

    model = create_model(seq_length, n_vocab, weights_file)
    # pick a random seed
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print("Алексей,")
    # generate characters
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\n\nУра!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Congratulates Alexey on his birthday')

    parser.add_argument('action', type=str,
                        choices=['train','congratulate'], help='what to do')

    parser.add_argument('--weights', type=str,
                        help='path to weights file')

    args = parser.parse_args()

    if args.action == 'train':
        train()
    else:
        if args.weights is None:
            print('please specify a weights file')
        generate_text(args.weights)

