import os
import numpy as np
import math
import operator
import pickle
import heapq
from sklearn.feature_extraction.text import CountVectorizer
from cnn_mm.config import config

def get_idx_from_sent(sent, word_idx_map, maxlen, padding):
    """
    mapping single raw sen to idx, last two element are: label and length
    :param sent:
    :param word_idx_map:
    :param maxlen:
    :param padding:
    :return:
    """
    x = []
    label = int(sent[-1])
    sent = sent[:-1]
    for i in range(padding):
        x.append(0)
    words = sent.split()
    for index, word in enumerate(words):
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < maxlen + 2 * padding:
        x.append(0)
    x.append(label)
    x.append(len(words))
    return x


def process_data(sents, word_idx_map, maxlen, padding):
    """process sentence list to idx"""
    data = [get_idx_from_sent(sent, word_idx_map, maxlen, padding) for sent in sents]
    return data


def get_keyword(count, smoothing_alpha, vocabulary, topk=30):
    """ NB to find important word"""
    count += smoothing_alpha

    ratio = []
    for index in range(count.shape[0]):
        class_count_ratio = 1.0 * count[index] / np.sum(count[index])
        other_count = np.sum(count, 0) - count[index]
        other_class_count_ratio = 1.0 * other_count / np.sum(other_count)
        num_count_scale = 1
        class_ratio = class_count_ratio * num_count_scale / other_class_count_ratio
        class_ratio = np.array(class_ratio)
        ratio.append(class_ratio)
    ratio = np.array(ratio)
    topk_ratio = [heapq.nlargest(topk, range(len(a)), a.take) for a in ratio]

    # topk_word = []
    # for l in topk_ratio:
    #     word = [vocabulary[i] for i in l]
    #     topk_word.append(word)

    topk_word = [vocabulary[w] for l in topk_ratio for w in l]
    # for w in topk_word:
    #     print(w)

    # ratio_max = np.max(ratio, axis=0)
    # beta = config['beta']
    # ratio_max_scale = (np.exp(beta * ratio_max) - 1) / (np.exp(beta * ratio_max) + 1)
    # count_word_score_come_from = np.argmax(ratio, axis=0)
    return topk_word


def my_process(train_sents, test_sents, word_idx_map, config,
               logpath, label_class, dev_sents=None):
    """
    split train Dev test data
    """
    np.random.shuffle(train_sents)
    if not dev_sents:
        # dev_sents = train_sents[int(len(train_sents) * 0.9):]
        # train_sents = train_sents[: int(len(train_sents) * 0.9)]
        dev_sents = []

    # all_data contain sentence specific by class
    all_data = [[] for _ in range(label_class)]
    label_ = [str(i) for i in range(label_class)]
    for sent in train_sents:
        index = label_.index(sent[-1])
        all_data[index].append(sent[:-1])

    all_data = [' '.join(s) for s in all_data]
    my_vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1)
    count = my_vectorizer.fit_transform(all_data).toarray()
    vocabulary = my_vectorizer.get_feature_names()

    topk_word = get_keyword(count, config['smoothing_alpha'], vocabulary, topk=config['topk'])

    topk_word_id = [word_idx_map[w] for w in topk_word]

    # for list in topk_word:
    #     ids = [word_idx_map[w] for w in list]
    #     topk_word_id.append(ids)

    # with open('{}/wordscore'.format(logpath), 'w') as f:
    #     k = len(topk_word)

    return train_sents, test_sents, dev_sents, topk_word, topk_word_id


def make_idx_data_cv(sentences, word_idx_map, cv, maxlen, padding, config, log_path):
    """
    # process datasets as 10-fold validation.
    :param sentences:
    :param word_idx_map:
    :param cv:
    :param maxlen:
    :param padding:
    :param config:
    :param log_path:
    :return:
    """
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    train_sents = []
    test_sents = []
    dev_sents = None
    label_set = set()

    if config['dataset'] in ['mr', 'subj', 'cr', 'mpqa']:
        for sen in sentences:
            # datum = {'y': label,
            #          'text': orig_rev,
            #          'num_words': len(orig_rev.split()),
            #          'split': np.random.randint(0, cv)}
            # s = get_idx_from_sent(sen['text'], word_idx_map, maxlen, padding)
            # s.append(sen['y'])
            sen_w = sen['text'] + str(sen['y'])
            label_set.add(sen['y'])
            if sen['split'] == cv:
                test_sents.append(sen_w)
            else:
                train_sents.append(sen_w)
    elif config['dataset'] in ['sst1', 'sst2']:
        dev_sents = []
        for sen in sentences:
            sen_w = sen['text'] + str(sen['y'])
            label_set.add(sen['y'])
            if sen['split'] == 'test':
                test_sents.append(sen_w)
            elif sen['split'] == 'train':
                train_sents.append(sen_w)
            elif sen['split'] == 'dev':
                dev_sents.append(sen_w)
    elif config['dataset'] in ['trec', 'ag_news']:
        for sen in sentences:
            sen_w = sen['text'] + str(sen['y'])
            label_set.add(sen['y'])
            if sen['split'] == 'test':
                test_sents.append(sen_w)
            elif sen['split'] == 'train':
                train_sents.append(sen_w)

    label_class = len(label_set)
    config['label_class'] = label_class
    for k, v in config.items():
        print(k, v)

    train_sents, test_sents, dev_sents, topk_word, topk_word_id = my_process(
        train_sents, test_sents, word_idx_map, config, log_path, label_class, dev_sents)

    train = process_data(train_sents, word_idx_map, maxlen, padding)
    test = process_data(test_sents, word_idx_map, maxlen, padding)
    dev = process_data(dev_sents, word_idx_map, maxlen, padding)

    train = np.asarray(train)
    test = np.asarray(test)
    dev = np.asarray(dev)

    return [train, test, dev], topk_word, topk_word_id


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    When used: batches = batch_iter(list(zip(train_x, train_y)), batch_size, epoch)
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            # If x is an integer, randomly permute np.arange(x).
            #  If x is an array, make a copy and shuffle the elements randomly.
            # np.random.permutation(10)
            # array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    data_path = '../data/' + config['dataset'] + '.p'
    print('Loading Data...')
    data_file = open(data_path, 'rb')
    x = pickle.load(data_file)
    data_file.close()
    sentences, W, W2, word_idx_map, vocab, maxlen = x[0], x[1], x[2], x[3], x[4], x[5]
    if config['debug'] or config['embedding'] == 'random':
        print('use random embedding')
        W = W2 # test in computer
    print('Data Loaded!')

    experiments_all = []
    experiments = 1 if config['debug'] else config['experiment']
    padding = 4
    config['config_str'] = 'test'
    for exp in range(experiments):
        final = []
        cv_num = 10 if (config['dataset'] in ['mr', 'subj', 'cr', 'mpqa']) and (not config['debug']) else 1
        for i in range(cv_num):
            log_path = 'logs/{}/experiment_{}/cv_{}'.format(config['config_str'], exp, i)
            datasets, topk_word, topk_word_id = make_idx_data_cv(
                sentences, word_idx_map, i, maxlen, padding, config, log_path)
            print('process done')

