import sys
import time
import tensorflow as tf
from util.process_keyword import *

np.random.seed(3306)


def softmaxY(Y, label_class):
    newY = []
    for y in Y:
        tmpY = [0] * label_class
        tmpY[y] = 1
        newY.append(tmpY)
    return np.asarray(newY)


def conv_weight_variable(shape):
    """# initialize W in CNN."""
    initial = np.random.uniform(-0.01, 0.01, shape)
    conv_W = tf.Variable(initial, name='conv_W', dtype=tf.float32)
    return conv_W


def conv_bias_variable(shape):
    """# initialize bias in CNN."""
    initial = tf.zeros(shape=shape)
    conv_b = tf.Variable(initial, name='conv_b', dtype=tf.float32)
    return conv_b


def fcl_weight_variable(shape, intial_type='normal'):
    """initialize W in fully connected layer."""
    if intial_type == 'uniform':
        initial = tf.random_uniform(-0.1, 0.1, shape)
    else:
        initial = tf.random_normal(shape=shape, stddev=0.01)
    fc_w = tf.Variable(initial, name='fcl_W')
    return fc_w


def fcl_bias_variable(shape):
    """initialize bias in fully connected layer."""
    initial = tf.zeros(shape=shape)
    fc_b = tf.Variable(initial, name='fcl_b')
    return fc_b


def conv1d(x, conv_W, conv_b):
    """compute convolution."""
    conv = tf.nn.conv1d(x, conv_W, stride=1, padding='SAME', name='conv')
    h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name='relu')
    return h


def max_pool(x):
    """ max-pooling."""
    return tf.reduce_max(x, axis=1)


class CNN_MM():
    def __init__(self, datasets, W, config, log_path, word_idx_map, maxlen, topk_word,
             embedding_dim=300, dropout=0.5, batch_size=50, nb_epoch=50, nb_filter=100,
             filter_length=[3, 4, 5], label_class=2, norm_lim=3, data_split=0):
        self.datasets = datasets
        self.W = W
        self.config = config
        self.log_path = log_path
        self.word_idx_map = word_idx_map
        self.max_len = maxlen
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.label_class = label_class
        self.norm_lim = norm_lim
        self.data_split = data_split
        self.topk_word = topk_word

        train, test, dev = self.load_data()
        self.x_train, self.y_train, self.x_train_len = train
        self.x_test, self.y_test, self.x_test_len = test
        self.x_dev, self.y_dev, self.x_dev_len = dev

        self.model = self.build_net()
        # Train
        self.saver = tf.train.Saver(max_to_keep=2)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())

    def load_data(self):
        """
        load train/test/valid data
        :return:
        """
        if config['debug']:
            self.datasets[0] = self.datasets[0][:202]
            self.datasets[1] = self.datasets[1][:202]
            self.datasets[2] = self.datasets[2][:202]
            permutation = np.arange(len(self.datasets[0]))
        else:
            permutation = np.random.permutation(len(self.datasets[0]))

        print('CV: ' + str(self.data_split + 1))

        x_train = np.asarray([d[:-2] for d in self.datasets[0]])
        y_train = np.asarray([d[-2] for d in self.datasets[0]])
        x_train_len = np.asarray([d[-1] for d in self.datasets[0]])

        x_test = np.asarray([d[:-2] for d in self.datasets[1]])
        y_test = np.asarray([d[-2] for d in self.datasets[1]])
        x_test_len = np.asarray([d[-1] for d in self.datasets[1]])

        x_dev = np.asarray([d[:-2] for d in self.datasets[2]])
        y_dev = np.asarray([d[-2] for d in self.datasets[2]])
        x_dev_len = np.asarray([d[-1] for d in self.datasets[2]])

        x_train = x_train[permutation]
        y_train = y_train[permutation]

        y_train = softmaxY(y_train, self.label_class)
        y_test = softmaxY(y_test, self.label_class)
        y_dev = softmaxY(y_dev, self.label_class)

        print('X_train shape:', x_train.shape)
        print('X_train_len shape:', x_train_len.shape)
        print('Y_train shape:', y_train.shape)

        print('X_dev shape:', x_dev.shape)
        print('Y_dev shape:', y_dev.shape)
        print('X_test shape:', x_test.shape)
        print('Y_test shape:', y_test.shape)

        return [x_train, y_train, x_train_len],  [x_test, y_test, x_test_len], [x_dev, y_dev, x_dev_len]

    def build_net(self):
        """
        build nn network
        :return:
        """
        # set all states to default.
        tf.reset_default_graph()
        tf.set_random_seed(1)
        keyword_num = self.config['topk'] * self.label_class
        # input.
        x = tf.placeholder(tf.int32, [None, self.max_len], name='input_x')
        y = tf.placeholder(tf.float32, [None, self.label_class], name='input_y')
        keyword = tf.placeholder(tf.int32, [keyword_num], name='keyword')
        keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # embedding.
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding_table = tf.Variable(self.W, name='embedding_table', trainable=True)
            embed_x = tf.nn.embedding_lookup(embedding_table, x)
            embed_keyword = tf.nn.embedding_lookup(embedding_table, keyword)
            print('embed_x={}, embed_key={}'.format(embed_x, embed_keyword))

        # build match matrix
        # spread b_l to dim(bs, l, k, d)
        # b_l = tf.expand_dims(embed_x, 3)
        # batch = tf.shape(x)[0]
        # ones_l = tf.ones((batch, self.max_len, 1, keyword_num), dtype=tf.float32)
        # l_sp = tf.transpose(tf.matmul(b_l, ones_l), [0, 1, 3, 2])
        # spread b_k to dim(bs, l, k, d)
        # b_k = tf.expand_dims(embed_keyword, 3)
        # ones_k = tf.ones((batch, keyword_num, 1, self.max_len), dtype=tf.float32)
        # k_sp = tf.transpose(tf.matmul(b_k, ones_k), [0, 3, 1, 2])
        # multiply two matrixs and sum at dim 3
        # match_matrix = tf.reduce_sum(tf.multiply(l_sp, k_sp), 3)
        sen_tile = tf.tile(tf.expand_dims(embed_x, 2), [1, 1, keyword_num, 1])  # (b, l, d) -> (b, l, k, d)
        print('sen_tile = {}'.format(sen_tile))
        match_matrix = tf.reduce_sum(tf.multiply(sen_tile, embed_keyword), 3)
        print(match_matrix)

        # CNN.
        pooled_outputs = []
        for i in self.filter_length:
            with tf.name_scope('conv_maxpool_%s' % i):
                filter_shape = [i, self.config['topk']*self.label_class, self.nb_filter]
                conv_W = conv_weight_variable(filter_shape)
                conv_b = conv_bias_variable([self.nb_filter])
                conv = conv1d(match_matrix, conv_W, conv_b)
                pooled = max_pool(conv)
                pooled_outputs.append(pooled)

        nb_filter_total = self.nb_filter * len(self.filter_length)
        h_pool = tf.concat(pooled_outputs, 1)
        h_pool_flat = tf.reshape(h_pool, [-1, nb_filter_total])

        # dropout.
        if config['use_dropout']:
            with tf.name_scope('dropout'):
                if config['dropouttrain']:  # test if dropout rate is trainable, bad result
                    # intial = np.random.uniform(0, 1, [nb_filter_total])
                    intial = np.ones([nb_filter_total])
                    intial *= 0
                    dropout_w = tf.Variable(intial)  # to be droppped out
                    dropout_w = tf.cast(dropout_w, tf.float32)
                    tmp_dis = tf.random_uniform([nb_filter_total], 0, 1)
                    tmp_ratio = tmp_dis / dropout_w  # <1 to be dropped out  <1 to be set to 0
                    mask_bool = tf.greater_equal(tmp_ratio, 1)
                    mask_w_fc = tf.cast(mask_bool, tf.float32)
                    h_pool_flat = h_pool_flat * mask_w_fc
                    # dropout_w_scale=0 in test
                    is_training = tf.less(keep_prob, 1)
                    is_training = tf.cast(is_training, tf.float32)
                    dropout_w_scale = dropout_w * is_training
                    h_pool_flat *= (1 / (1 - dropout_w_scale))

                    # in training
                    # h_pool_flat = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
                else:
                    if config['magic_scale']:  # magic_scale is bad in this
                        h_pool_flat = tf.nn.dropout(
                            h_pool_flat, keep_prob) * (1 - keep_prob + self.dropout)
                    else:
                        h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob)

        # fully connected layer.
        with tf.name_scope('fcl'):
            fc_W = fcl_weight_variable([nb_filter_total, self.label_class])
            fc_b = fcl_bias_variable([self.label_class])
            fc_output = tf.matmul(h_pool_flat, fc_W) + fc_b
            y_pred = tf.nn.softmax(fc_output)
            y_pred = tf.clip_by_value(y_pred, clip_value_min=1e-6, clip_value_max=1 - 1e-6)

        cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(y * tf.log(y_pred) + (1 - y) * tf.log(1 - y_pred),
                           reduction_indices=[1]))

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-08)
        # optimizer = tf.train.AdamOptimizer(learning_rate=1.0)
        gvs = optimizer.compute_gradients(cross_entropy)
        capped_gvs = [((tf.clip_by_norm(grad, self.norm_lim)), var) for grad, var in gvs]
        train_step = optimizer.apply_gradients(capped_gvs)
        prediction = tf.arg_max(y, 1)
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print('trainable variables are ...')
        ws = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
        for v in ws:
            print(v.name)

        return {'x': x, 'y': y, 'keyword': keyword, 'keep_prob': keep_prob,
                'cross_entropy': cross_entropy, 'optimizer': optimizer,
                'train_step': train_step, 'prediction': prediction,
                'correct_prediction': correct_prediction, 'acc': acc}

    def train_cv(self):
        """

        :return:
        """
        print('start train')
        best_accuracy = 0
        best_loss = 100
        time_delay = 0

        # train_acc = 0
        # train_loss = 0
        test_acc = 0
        # test_loss = 0
        # dev_acc = 0
        # dev_loss = 0

        for e in range(self.nb_epoch):
            if time_delay == config['timedelay']:
                break
            epoch_starttime = time.time()
            i = 0

            train_acc_list = []
            train_loss_list = []
            train_batch = batch_iter(list(zip(self.x_train, self.y_train)), batch_size=self.batch_size, num_epochs=1,
                                   shuffle=True)
            for train in train_batch:
                x_train, y_train = zip(*train)
                train_result = [self.model['train_step'], self.model['acc'], self.model['cross_entropy']]
                _, temp_acc, temp_loss = self.sess.run(train_result, feed_dict={
                    self.model['x']: x_train, self.model['y']: y_train,
                    self.model['keyword']: self.topk_word,
                    self.model['keep_prob']: self.dropout})
                train_acc_list.append(temp_acc)
                train_loss_list.append(temp_loss)
            # while i + 1 < len(self.x_train):
            #     if i + self.batch_size < len(self.x_train):
            #         batch_xs = self.x_train[i:i + self.batch_size]
            #         batch_x_l = self.x_train_len[i:i+self.batch_size]
            #         batch_ys = self.y_train[i:i + self.batch_size]
            #     else:
            #         batch_xs = self.x_train[i:]
            #         batch_x_l = self.x_train_len[i:]
            #         batch_ys = self.y_train[i:]
            #     i += self.batch_size
            #
            #     temp_batch_size = len(batch_xs)
            #     _, temp_acc, temp_loss = self.sess.run(
            #         [self.model['train_step'], self.model['acc'], self.model['cross_entropy']], feed_dict={
            #         self.model['x']: batch_xs, self.model['y']: batch_ys,
            #         self.model['keyword']: [self.topk_word],
            #         self.model['keep_prob']: self.dropout})
            #     train_acc_list.append(temp_acc)
            #     train_loss_list.append(temp_loss)

            train_acc = sum(train_acc_list)/len(train_acc_list)
            train_loss = sum(train_loss_list) / len(train_loss_list)

            if len(self.x_dev) != 0:
                # dev batch
                dev_batch = batch_iter(list(zip(self.x_dev, self.y_dev)), batch_size=self.batch_size, num_epochs=1, shuffle=False)
                dev_acc_list = []
                dev_loss_list = []
                for dev in dev_batch:
                    x_dev, y_dev = zip(*dev)
                    dev_result = [self.model['acc'], self.model['cross_entropy']]
                    temp_acc, temp_loss = self.sess.run(dev_result, feed_dict={
                        self.model['x']: x_dev, self.model['y']: y_dev,
                        self.model['keyword']: self.topk_word,
                        self.model['keep_prob']: 1.0})
                    dev_acc_list.append(temp_acc)
                    dev_loss_list.append(temp_loss)
                dev_acc = sum(dev_acc_list)/len(dev_acc_list)
                dev_loss = sum(dev_loss_list) / len(dev_loss_list)

                # test batch
                test_batch = batch_iter(list(zip(self.x_test, self.y_test)), batch_size=self.batch_size, num_epochs=1, shuffle=False)
                test_acc_list = []
                if dev_acc > best_accuracy or dev_loss < best_loss:
                    time_delay = 0
                    best_accuracy = dev_acc
                    best_loss = dev_loss
                    self.saver.save(self.sess, self.log_path + '/model.ckpt')

                    # test batch
                    for test in test_batch:
                        x_test, y_test = zip(*test)
                        temp_acc, test_prediction = self.sess.run(
                            [self.model['acc'], self.model['prediction']],
                            feed_dict={self.model['x']: x_test, self.model['y']: y_test,
                                       self.model['keyword']: self.topk_word,
                                       self.model['keep_prob']: 1.0})
                        test_acc_list.append(temp_acc)

                    test_acc = sum(test_acc_list) / len(test_acc_list)

                else:
                    time_delay += 1

                # break
                # dev_loss = '_'.join([str(i)[:6] for i in list(dev_loss)])
                sys.stdout.write('Epoch: %d' % (e + 1))
                sys.stdout.write(' Train acc: %.5f' % train_acc)
                sys.stdout.write(' Loss: %.5f' % train_loss)
                sys.stdout.write('\tDev Loss: %.5f'% dev_loss)
                sys.stdout.write('Acc: %.6f' % dev_acc)
                sys.stdout.write('\tTest Acc: %.6f' % test_acc)
                sys.stdout.write('\tTime: %.1fs' % (time.time() - epoch_starttime))
                sys.stdout.write(' Timedelay: %.1fs' % time_delay)
                sys.stdout.write('\n')

            else:
                # test batch
                test_batch = batch_iter(
                    list(zip(self.x_test, self.y_test)), batch_size=self.batch_size,
                    num_epochs=1, shuffle=False)

                # test batch
                test_acc_list = []
                test_loss_list = []
                for test in test_batch:
                    x_test, y_test = zip(*test)
                    temp_acc, temp_loss, test_prediction = self.sess.run(
                        [self.model['acc'], self.model['cross_entropy'],
                         self.model['prediction']],
                        feed_dict={self.model['x']: x_test, self.model['y']: y_test,
                                    self.model['keyword']: self.topk_word,
                                    self.model['keep_prob']: 1.0})
                    test_acc_list.append(temp_acc)
                    test_loss_list.append(temp_loss)

                test_acc = sum(test_acc_list) / len(test_acc_list)
                test_loss = sum(test_loss_list) / len(test_loss_list)

                if test_acc > best_accuracy or test_loss < best_loss:
                    time_delay = 0
                    best_accuracy = test_acc
                    best_loss = test_loss
                    self.saver.save(self.sess, self.log_path + '/model.ckpt')
                else:
                    time_delay += 1

                sys.stdout.write('Epoch:%d' % (e + 1))
                sys.stdout.write(' Train acc:%.5f' % train_acc)
                sys.stdout.write(' Loss:%.5f' % train_loss)
                sys.stdout.write('\tTest Acc:%.5f' % test_acc)
                sys.stdout.write(' loss: %.5f' % test_loss)
                sys.stdout.write('\tTime: %.1fs' % (time.time() - epoch_starttime))
                sys.stdout.write(' Timedelay: %.1fs' % time_delay)
                sys.stdout.write('\n')

        # Test trained model.
        # f_wrong = open('{}/test_wrong'.format(self.log_path), 'w')
        # f_right = open('{}/test_right'.format(self.log_path), 'w')
        # [1, 0] neg
        # [0, 1] pos
        # id_word_map = {j: i for i, j in self.word_idx_map.items()}
        # id_word_map[0] = 'padding'
        # k = 3
        # for index, pred in enumerate(test_prediction):
        #     sent = ' '.join([id_word_map[i] for i in self.x_test[index] if i > 0])
        #     sent_score = W_score_ref[X_test[index]]
        #     top_k = sent_score.argsort()[(-1) * k:][::-1]
        #     key_word_str = ''
        #     label = np.argmax(self.y_test[index])
        #
        #     for mm in top_k:
        #         word_id = self.x_test[index][mm]
        #
        #         ratio_divide = (W_score[word_id] - 1)
        #         use_ratio = ratio_divide
        #
        #         if word_id in w_score_come_from:
        #             if label == w_score_come_from[word_id]:
        #                 key_word_str += '{} {} {} {} {}\t'.format(
        # id_word_map[word_id], W_score_ref[word_id], ratio_divide, use_ratio, count_ref[word_id])
        #     if not key_word_str:
        #         key_word_str = 'no good mask'
        #
        #     if config['dataset'] not in ['sst1', 'trec']:
        #         pred = ['neg', 'pos'][pred]
        #         label = ['neg', 'pos'][label]
        #     if pred == label:
        #         f_right.write('{}\n{}\nref: {}\n\n'.format(sent, pred, key_word_str))
        #     else:
        #         f_wrong.write('{}\npred: {} label: {}\nref: {}\n\n'.format(sent, pred, label, key_word_str))
        #
        # f_wrong.close()
        # f_right.close()

        print('CV: ' + str(self.data_split + 1) + ' Test Accuracy: %.4f%%\n' %
              (100 * best_accuracy))
        self.sess.close()

        return best_accuracy


def main(args, para=None):
    """
    main function
    :param args:
    :param para:
    :return:
    """
    from cnn_mm import config
    config = config.config
    config['baseline_same_drop'] = False
    config['dropouttrain'] = False

    if args.debug is not None:
        config['debug'] = args.debug
    if args.dataset:
        config['dataset'] = args.dataset
    if args.baseline:
        config['mask_ratio'] = False
    if args.nodropout:
        config['use_dropout'] = False
    if args.dropouttrain:
        config['dropout_train'] = True
    if args.experiment:
        config['experiment'] = args.experiment
    if args.batchsize:
        config['batch_size'] = args.batchsize
    if args.beta:
        config['beta'] = args.beta
    if args.magicscale:
        config['magic_scale'] = True
    if args.random:
        config['embedding'] = 'random'
    if args.same_drop_prob:
        config['same_drop_prob'] = args.same_drop_prob
    if args.topk:
        config['topk'] = args.topk

    data_path = '../data/' + config['dataset'] + '.p'

    config['config_str'] = '{}/{}/top{}/'.format(config['dataset'], 'cnn', config['topk'])

    # if config['dropout_train']:
    #     config['config_str'] += 'dropout_train'
    # else:
    config['config_str'] += 'dropout_{}_magic_{}'.format(config['use_dropout'], config['magic_scale'])

    config['config_str'] += '_{}'.format(config['embedding'])
    # print(config)
    # exit()

    padding = 4
    # padding = 0
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

    for exp in range(experiments):
        final = []
        cv_num = 10 if (config['dataset'] in ['mr', 'subj', 'cr', 'mpqa']) and (not config['debug']) else 1
        for i in range(cv_num):
            log_path = 'logs/{}/experiment_{}/cv_{}'.format(config['config_str'], exp, i)
            datasets, topk_word, topk_word_id = make_idx_data_cv(
                sentences, word_idx_map, i, maxlen, padding, config, log_path)
            print('process done')

            model = CNN_MM(
                datasets, W, config, log_path, word_idx_map, topk_word=topk_word_id,
                data_split=i, maxlen=maxlen + 2 * padding, label_class=config['label_class'],
                batch_size=config['batch_size'])
            acc = model.train_cv()
            final.append(acc)

        final = np.mean(final)
        experiments_all.append(final)
        print('Experiment {} Final Test Accuracy: {}'.format(exp, final))
    f_model = open('result', 'a')
    print('\n\nAll Experiment Final Test Accuracy: {}'.format(experiments_all))
    print('All experiments final mean : {}'.format(np.mean(experiments_all)))
    f_model.write('\n')
    f_model.write(config['config_str'])
    f_model.write('\nAll Experiment Final Test Accuracy: {}'.format(experiments_all))
    f_model.write('\nAll experiments final mean : {}\n\n'.format(np.mean(experiments_all)))


# entry point.
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # store_true. default False, set to true when input something
    parser.add_argument("--debug", help="default false", action='store_true')
    parser.add_argument("--dataset", help="datset to choose", type=str)
    parser.add_argument("--baseline", help="default false, cnn baseline", action='store_true')
    parser.add_argument("--baseline_same_drop", help="in dropout layer, all words dropped according to the same probability, default false", action='store_true')
    parser.add_argument("--dropouttrain", help="in fully connected layer, every w has a trained dropout probability, and are masked in this probability, default false", action='store_true')
    parser.add_argument("--nodropout", help="default false", action='store_true')
    parser.add_argument("--magicscale", help="default false", action='store_true')
    parser.add_argument("--random", help="default false", action='store_true')
    parser.add_argument("--experiment", help="experiment number", type=int)
    parser.add_argument("--beta", help="beta 0.01 0.1", type=float)
    parser.add_argument("--batchsize", help="batch size", type=int)
    parser.add_argument("--topk", help="top k keyword", type=int)
    parser.add_argument("--same_drop_prob", help="same_drop_prob in baseline_same_drop", type=float)

    args = parser.parse_args()
    print(args)
    main(args)
