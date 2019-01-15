import pickle
import re
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
# from docopt import docopt
import argparse
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.utils import shuffle
from concurrent import futures
from cnn_mm.config import config

np.random.seed(3306)


def build_data_cv(filename, sentences, vocab, clean_string=True, label_file=None, is_cv=False):
    """
    loads data and split into 10 folds.
    """
    # with open(filename, 'r', encoding='ISO-8859-1') as f:
    with open(filename, 'r') as f:
        for index, line in enumerate(f):
            if is_cv:
                split_ = np.random.randint(0, 10)
                label = label_file
                rev = [line.strip()]
            else:
                if 'train' in filename:
                    split_ = 'train'
                elif 'test' in filename:
                    split_ = 'test'
                elif 'dev' in filename:
                    split_ = 'dev'
                else:
                    print('wrong')
                # print(line)
                if '.csv' in filename:  # ag_news  label [1,2,3,4]
                    label = int(line[1]) - 1
                    rev = [line.strip()[4:]]

                else:
                    label = int(line[0])
                    rev = [line.strip()[2:]]

            if clean_string:
                orig_rev = clean_str(' '.join(rev))
            else:
                orig_rev = ' '.join(rev).lower()

            # # print(label, rev)
            # print(orig_rev)
            # if index == 0:
            #     exit()

            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1

            datum = {'y': label,
                     'text': orig_rev,
                     'num_words': len(orig_rev.split()),
                     'split': split_}
            # print(datum)
            # exit()
            sentences.append(datum)


def get_W(word_vecs, k=300):
    """
    get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    loads 300x1 word vectors from file.
    """
    word_vecs = {}
    with open(fname, 'rb') as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word)
                    break
                if ch != b'\n':
                    word.append(ch)
            # word = str(word, 'UTF-8')
            word = str(word)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    add random vectors of unknown words which are not in pre-trained vector file.
    if pre-trained vectors are not used, then initialize all words in vocab with random value.
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def clean_str(string):
    """
    clean data
    """
    string = re.sub(r'[^A-Za-z0-9(),!?\'`]', ' ', string)
    string = re.sub(r'\'s', ' \'s', string) 
    string = re.sub(r'\'ve', ' \'ve', string) 
    string = re.sub(r'n\'t', ' n\'t', string) 
    string = re.sub(r'\'re', ' \'re', string) 
    string = re.sub(r'\'d', ' \'d', string) 
    string = re.sub(r'\'ll', ' \'ll', string) 
    string = re.sub(r',', ' , ', string) 
    string = re.sub(r'!', ' ! ', string) 
    string = re.sub(r'\(', ' \( ', string) 
    string = re.sub(r'\)', ' \) ', string) 
    string = re.sub(r'\?', ' \? ', string) 
    string = re.sub(r'\s{2,}', ' ', string)    
    return string.strip().lower()


def clean_str_ag(string):
    """clean data"""
    # string = re.sub(r'\\\'', '', string)

    string = re.sub(r'[^A-Za-z0-9(),!?\'`]', ' ', string)
    string = re.sub(r'\'s', ' \'s', string)
    string = re.sub(r'\'ve', ' \'ve', string)
    string = re.sub(r'n\'t', ' n\'t', string)
    string = re.sub(r'\'re', ' \'re', string)
    string = re.sub(r'\'d', ' \'d', string)
    string = re.sub(r'\'ll', ' \'ll', string)
    string = re.sub(r',', ' , ', string)
    string = re.sub(r'!', ' ! ', string)
    string = re.sub(r'\(', ' \( ', string)
    string = re.sub(r'\)', ' \) ', string)
    string = re.sub(r'\?', ' \? ', string)
    string = re.sub(r'\s{2,}', ' ', string)

    return string.strip().lower()


# main function.
def process(dataset, is_cv):
    """
    main function
    :param dataset:
    :param is_cv:
    :return: data.p
    """
    print('\n\nprocess data {}'.format(dataset))

    # vectors_file = args['<vectors_file>']                   # pre-trained word vectors file
    # data_folder = ['rt-polarity.neg', 'rt-polarity.pos']    # data files

    dataset = os.path.join('../data', dataset)
    datafile = dataset + '.p'

    data_folder = sorted(os.listdir(dataset))
    print(data_folder)
    data_folder = [os.path.join(dataset, file) for file in data_folder]
    print('Loading Data...')
    sentences = []              # sentences processed
    vocab = defaultdict(float)  # vocabulary
    # process data
    if is_cv:
        for index, file in enumerate(data_folder):
            build_data_cv(file, sentences, vocab, is_cv=True, label_file=index, clean_string=True)
    else:
        for index, file in enumerate(data_folder):
            build_data_cv(file, sentences, vocab, is_cv=False, clean_string=True)

    # print(sorted(vocab.items(), key=operator.itemgetter(1), reverse=True))

    maxlen = np.max(pd.DataFrame(sentences)['num_words'])    # max length of sentences
    print('Data Loaded!')
    print('Number Of Sentences: ' + str(len(sentences)))
    print('Vocab Size: ' + str(len(vocab)))
    print('Max Sentence Length: ' + str(maxlen))

    print('Loading Vectors...')
    rand_vecs = {}  # random vectors of all words
    if not debug:
        vectors = load_bin_vec(vectors_file, vocab)     # pre-trained vectors
        print('Vectors Loaded!')
        print('Words Already In Vectors: ' + str(len(vectors)))
        #add random vectors of words which are not in vocab.
        add_unknown_words(vectors, vocab)
        # the W used and the word_idx_map must come from the same function get_W
        W, word_idx_map = get_W(vectors)    # vectors of all words and a map of words to ids
        W2, _= get_W(rand_vecs)            # random vectors of all words which are related to ids

    else:
        W = 0
        add_unknown_words(rand_vecs, vocab)
        W2, word_idx_map= get_W(rand_vecs)            # random vectors of all words which are related to ids

    # save sentences and vectors
    pickle.dump([sentences, W, W2, word_idx_map, vocab, maxlen], open(datafile, 'wb'))
    print('Dataset created!')


# entry point.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_path", help="GoogleNews vector path", default='', type=str)
    args = parser.parse_args()

    vectors_file = args.vector_path if args.vector_path else 'GoogleNews-vectors-negative300.bin'
    if not os.path.exists(vectors_file):
        print('vector path {} does not exist'.format(vectors_file))
        exit()
    dataset_cv = ['cr', 'mr', 'subj', 'mpqa']
    # some wrong in subj, trec  fix by add encoding=ISO-8859-1
    debug = False
    # debug = True
    dataset_no_cv = ['sst1', 'sst2', 'trec']
    # dataset_no_cv = ['trec']
    for data in dataset_cv:
        process(data, is_cv=True)

    for data in dataset_no_cv:
        process(data, is_cv=False)

    # data_new = 'ag_news'
    # process(data_new, is_cv=False)








