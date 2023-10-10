import torch
import gensim
import numpy as np
import jieba
import pickle
import os
import json
import random
from collections import Counter


class IntegerIndex:
    def __init__(self, index):
        self.index = index


cw = lambda x: list(jieba.cut(x))


def data_load(corpus, mode='train'):
    if os.path.exists("../file/{}.pkl".format(mode)):
        pkl_file = open("../file/{}.pkl".format(mode), 'rb')
        data_list = pickle.load(pkl_file)
    else:
        text = open('../file/{}.tsv'.format(mode), 'r', encoding='utf-8').readlines()
        data = [[line.split('\t')[0], cw(line.split('\t')[1].strip())] for line in text[1:]]
        output = open('../file/{}.pkl'.format(mode), 'wb')
        data_list = []
        for l, s in data:
            data_list.append((int(l), s, ''.join(s)))
        pickle.dump(data_list, output, -1)
        output.close()

    if os.path.exists('../file/TNT_{}_{}.pkl'.format(corpus, mode)):
        pkl_file = open("../file/TNT_{}_{}.pkl".format(corpus, mode), 'rb')
        f_data = pickle.load(pkl_file)
    else:
        model_word = gensim.models.Word2Vec.load('../file/word2vec_{}'.format(corpus))
        model_char = gensim.models.Word2Vec.load('../file/word2vec_wiki')
        vocab_word = model_word.wv.vocab
        vocab_char = model_char.wv.vocab
        f_data = []
        for l, s, s_raw in data_list:
            f_data.append((l, [vocab_word.get(word, IntegerIndex(-1)).index for word in s], [vocab_char.get(c, IntegerIndex(-1)).index for c in s_raw], s_raw))
        output = open('../file/TNT_{}_{}.pkl'.format(corpus, mode), 'wb')
        pickle.dump(f_data, output, -1)
        output.close()

    return f_data


def data_preprocess(corpus, mode='train', val_ratio=0.2):
    pkl_file = open('../file/label2ids_CNT.pkl', 'rb')
    label2ids = pickle.load(pkl_file)
    if mode == 'test':
        if os.path.exists("../file/CNT_{}.pkl".format(mode)):
            pkl_file = open("../file/CNT_{}.pkl".format(mode), 'rb')
            data_list = pickle.load(pkl_file)
        else:
            txt = open('../file/{}_file.txt'.format(mode)).readlines()
            data = [[label2ids[line.split('\t')[0]], cw(line.split('\t')[1].strip())] for line in txt]
            output = open('../file/CNT_{}.pkl'.format(mode), 'wb')
            data_list = []
            for l, s in data:
                data_list.append((int(l), s, ''.join(s)))
            pickle.dump(data_list, output, -1)
            output.close()

        if os.path.exists('../file/CNT_{}_{}.pkl'.format(corpus, mode)):
            pkl_file = open("../file/CNT_{}_{}.pkl".format(corpus, mode), 'rb')
            f_data = pickle.load(pkl_file)
        else:
            model_word = gensim.models.Word2Vec.load('../file/word2vec_{}'.format(corpus))
            model_char = gensim.models.Word2Vec.load('../file/word2vec_wiki')
            vocab_word = model_word.wv.vocab
            vocab_char = model_char.wv.vocab
            f_data = []
            for l, s, s_raw in data_list:
                f_data.append((l, [vocab_word.get(word, IntegerIndex(-1)).index for word in s],
                               [vocab_char.get(c, IntegerIndex(-1)).index for c in s_raw], s_raw))
            output = open('../file/CNT_{}_{}.pkl'.format(corpus, mode), 'wb')
            pickle.dump(f_data, output, -1)
            output.close()
        return f_data
    else:
        mode = 'train'
        if os.path.exists("../file/CNT_{}.pkl".format(mode)):
            pkl_file = open("../file/CNT_{}.pkl".format(mode), 'rb')
            data_list = pickle.load(pkl_file)
        else:
            txt = open('../file/{}_file.txt'.format(mode)).readlines()
            data = [[label2ids[line.split('\t')[0]], cw(line.split('\t')[1].strip())] for line in txt]
            output = open('../file/CNT_{}.pkl'.format(mode), 'wb')
            data_list = []
            for l, s in data:
                data_list.append((int(l), s, ''.join(s)))
            pickle.dump(data_list, output, -1)
            output.close()

        if os.path.exists('../file/CNT_{}_{}.pkl'.format(corpus, mode)):
            pkl_file = open("../file/CNT_{}_{}.pkl".format(corpus, mode), 'rb')
            f_data = pickle.load(pkl_file)
        else:
            model_word = gensim.models.Word2Vec.load('../file/word2vec_{}'.format(corpus))
            model_char = gensim.models.Word2Vec.load('../file/word2vec_wiki')
            vocab_word = model_word.wv.vocab
            vocab_char = model_char.wv.vocab
            f_data = []
            for l, s, s_raw in data_list:
                f_data.append((l, [vocab_word.get(word, IntegerIndex(-1)).index for word in s],
                               [vocab_char.get(c, IntegerIndex(-1)).index for c in s_raw], s_raw))
            output = open('../file/CNT_{}_{}.pkl'.format(corpus, mode), 'wb')
            pickle.dump(f_data, output, -1)
            output.close()
        targets = np.array([l[0] for l in f_data])
        train_ids, test_ids = class_balance_sampler(targets, num_classes=len(label2ids), test_ratio=val_ratio)
        assert len(test_ids) + len(train_ids) == len(f_data)
        return [f_data[i] for i in train_ids], [f_data[i] for i in test_ids]


def class_balance_sampler(targets, num_classes, test_ratio):
    train_ids = []
    test_ids = []
    for i in range(num_classes):
        ids = np.where(targets == i)[0]
        total_num = len(ids)
        tr_num = int(total_num * (1 - test_ratio))
        np.random.shuffle(ids)
        tr_ids = ids[:tr_num]
        te_ids = ids[tr_num:]
        train_ids.append(tr_ids)
        test_ids.append(te_ids)
    return np.concatenate(train_ids, axis=0), np.concatenate(test_ids, axis=0)


def FCT_preprocess(corpus, mode='train', val_ratio=0.1, max_len=512):

    if os.path.exists('../file/FCT_{}_{}.pkl'.format(corpus, mode)):
        pkl_file = open("../file/FCT_{}_{}.pkl".format(corpus, mode), 'rb')
        f_data = pickle.load(pkl_file)
    else:
        if not os.path.exists('../file/FCT/label_mapping.pkl') or not os.path.exists('../file/FCT/file_name.pkl'):
            file_names = []
            for file in os.listdir('../file/FCT/test'):
                if os.path.isdir('../file/FCT/test'+'/{}'.format(file)):
                    file_names.append(file)
            label_mapping = {name: i for i, name in enumerate(file_names)}
            output_file_name = open('../file/FCT/file_name.pkl', 'wb')
            output_label_mapping = open('../file/FCT/label_mapping.pkl', 'wb')
            pickle.dump(file_names, output_file_name, -1)
            pickle.dump(label_mapping, output_label_mapping, -1)
            output_file_name.close()
            output_label_mapping.close()
        else:
            file_names = open('../file/FCT/file_name.pkl', 'rb')
            file_names = pickle.load(file_names)
            label_mapping = open('../file/FCT/label_mapping.pkl', 'rb')
            label_mapping = pickle.load(label_mapping)
        assert len(file_names) == 20
        model_word = gensim.models.Word2Vec.load('../file/word2vec_{}'.format(corpus))
        model_char = gensim.models.Word2Vec.load('../file/word2vec_wiki')
        vocab_word = model_word.wv.vocab
        vocab_char = model_char.wv.vocab
        f_data = []
        for file_name in file_names:
            for sample_path in os.listdir('../file/FCT/{}/{}/utf8'.format(mode, file_name)):
                assert file_name == sample_path[:len(file_name)]
                sample = open('../file/FCT/{}/{}/utf8/{}'.format(mode, file_name, sample_path), 'r', encoding='utf-8')
                line_list = sample.readlines()
                content = sample.read()
                if len(line_list) < 1:
                    continue
                text = ''
                start = False
                for line in line_list:
                    if '】' not in content and '【' not in content:
                        start = True
                    if start:
                        text += line.strip()
                    if '正  文' in line:
                        start = True
                assert len(text) > 0
                s = cw(text)
                f_data.append((label_mapping[file_name], [vocab_word.get(word, IntegerIndex(-1)).index for word in s],
                               [vocab_char.get(c, IntegerIndex(-1)).index for c in text], text))
        output = open('../file/FCT_{}_{}.pkl'.format(corpus, mode), 'wb')
        pickle.dump(f_data, output, -1)
        output.close()

    if mode == 'train':
        targets = np.array([l[0] for l in f_data])
        train_ids, test_ids = class_balance_sampler(targets, num_classes=20, test_ratio=val_ratio)
        assert len(test_ids) + len(train_ids) == len(f_data)
        return [f_data[i] for i in train_ids], [f_data[i] for i in test_ids]

    return f_data


def data_description(f_data, dataset='CNT'):
    length = [len(f[-1]) for f in f_data]
    labels = [int(f[0]) for f in f_data]
    freqs = np.array([num / len(length) for num in list(Counter(labels))])

    # print('{}. len: {}, min:{}, max: {}, Avg.: {}'.format(dataset, len(length), min(length), max(length), sum(length) // len(length)))
    print('{} {}& ({}, {})'.format(dataset, len(length),sum(length) // len(length), max(length)))


if __name__ == '__main__':
    # CNT
    train_list, val_list = data_preprocess('wx', 'train', 0.1)


