import numpy as np
import torch.nn as nn
import torch
import os
import random
from collections import Counter
from sklearn.decomposition import PCA
import warnings
import time
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
warnings.filterwarnings('ignore')


np.random.seed(1)
torch.manual_seed(1)
random.seed(1)


def normalize(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def normalize_l(vecs):
    mean = vecs.mean(axis=0, keepdims=True)
    std = vecs.std(axis=0, keepdims=True)
    return (vecs - mean) / std


def compute_kernel_bias(vecs, variance_ratio=1.01):
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)

    u, s, vh = np.linalg.svd(cov)
    W = u
    ratio = 0
    variance_ratio_ = s / s.sum()
    for n_components, r in enumerate(variance_ratio_):
        ratio += r
        if ratio >= variance_ratio:
            break
    print(n_components)
    args.log_file.write('{} '.format(n_components))
    args.log_file.flush()
    W = W[:, :n_components+1]
    return W, -mu


def transform_and_normalize(vecs, kernel, bias):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def train_one_epoch(model, dataloader, optimizer, loss_func, lam=0):
    model.train()
    loss_mean = 0
    for d in dataloader:
        feature, label = d[:, :-1], d[:, -1]
        label = label.long().cuda()
        out = model(feature.cuda())
        loss = loss_func(out, label)
        loss += lam * torch.abs(model.weight).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_mean += loss.item()
    loss_mean = loss_mean / len(dataloader)
    return loss_mean


def test(model, dataloader):
    total_num = 0
    correct_num = 0
    pred_list = []
    label_list = []
    model.eval()
    for d in dataloader:
        feature, label = d[:, :-1], d[:, -1]
        out = model(feature.cuda())
        pred = out.argmax(1).cpu()

        correct_num += pred.float().eq(label.float()).sum().numpy()
        total_num += len(label)
        pred_list.extend(list(pred.detach().cpu().numpy().astype('int64')))
        label_list.extend(list(label.detach().cpu().numpy().astype('int64')))
    assert len(label_list) == len(pred_list)
    acc = correct_num / (total_num + 1e-8)
    f1_macro = f1_score(label_list, pred_list, labels=[i for i in range(args.num_classes)], average="macro")
    recall_macro = recall_score(label_list, pred_list, labels=[i for i in range(args.num_classes)], average="macro")
    precision_macro = precision_score(label_list, pred_list, labels=[i for i in range(args.num_classes)], average="macro")

    log_str = 'Accuracy={:.4f}. Precision={:.4f}. Recall={:.4f}. F1={:.4f}'.format(
        acc,
        precision_macro,
        recall_macro,
        f1_macro
    )
    model.train()
    return f1_macro, log_str


def calculate_sample_weight(labels):
    labels = [int(d) for d in labels]
    freq = Counter(labels)
    weights = torch.Tensor([freq[l] for l in labels])
    return 1. / weights


def load_data(dataset, word=None, char=None, pinyin=None, glyph=None, corpus='wx', seed=10):
    train_features = np.load("../log/{}/features/{}_{}_char:{}_word:{}_pinyin:{}_glyph:{}_avg:False_train_set_features.npy".format(
        dataset, seed, corpus, char, word, pinyin, glyph
    ))
    train_labels = np.load("../log/{}/features/{}_{}_char:{}_word:{}_pinyin:{}_glyph:{}_avg:False_train_set_labels.npy".format(
        dataset, seed, corpus, char, word, pinyin, glyph
    ))

    val_features = np.load("../log/{}/features/{}_{}_char:{}_word:{}_pinyin:{}_glyph:{}_avg:False_val_set_features.npy".format(
        dataset, seed, corpus, char, word, pinyin, glyph
    ))
    val_labels = np.load("../log/{}/features/{}_{}_char:{}_word:{}_pinyin:{}_glyph:{}_avg:False_val_set_labels.npy".format(
        dataset, seed, corpus, char, word, pinyin, glyph
    ))
    test_features = np.load("../log/{}/features/{}_{}_char:{}_word:{}_pinyin:{}_glyph:{}_avg:False_test_set_features.npy".format(
        dataset, seed, corpus, char, word, pinyin, glyph
    ))
    test_labels = np.load("../log/{}/features/{}_{}_char:{}_word:{}_pinyin:{}_glyph:{}_avg:False_test_set_labels.npy".format(
        dataset, seed, corpus, char, word, pinyin, glyph
    ))
    return [train_features, train_labels], [val_features, val_labels], [test_features, test_labels]


def train(args):

    all_granularity_features = []
    all_granularity_labels = []
    variance_ratios = []
    granularity_str = 'use '
    if args.word is not None:
        word_train, word_val, word_test = load_data(dataset=args.dset, word=args.word, corpus=args.corpus, seed=args.seed)
        all_granularity_features.append([word_train[0], word_val[0], word_test[0]])
        all_granularity_labels.append([word_train[1], word_val[1], word_test[1]])
        variance_ratios.append(args.variance_ratio_word)
        granularity_str += 'word, '
        args.log_file.write('word ')
        args.log_file.flush()

    if args.char is not None:
        char_train, char_val, char_test = load_data(dataset=args.dset, char=args.char, corpus=args.corpus, seed=args.seed)
        all_granularity_features.append([char_train[0], char_val[0], char_test[0]])
        all_granularity_labels.append([char_train[1], char_val[1], char_test[1]])
        variance_ratios.append(args.variance_ratio_char)
        granularity_str += 'character, '
        args.log_file.write('character ')
        args.log_file.flush()

    if args.pinyin is not None:
        pinyin_train, pinyin_val, pinyin_test = load_data(dataset=args.dset, pinyin=args.pinyin, corpus=args.corpus, seed=args.seed)
        all_granularity_features.append([pinyin_train[0], pinyin_val[0], pinyin_test[0]])
        all_granularity_labels.append([pinyin_train[1], pinyin_val[1], pinyin_test[1]])
        variance_ratios.append(args.variance_ratio_pinyin)
        granularity_str += 'pinyin, '
        args.log_file.write('pinyin ')
        args.log_file.flush()

    if args.glyph is not None:
        glyph_train, glyph_val, glyph_test = load_data(dataset=args.dset, glyph=args.glyph, corpus=args.corpus, seed=args.seed)
        all_granularity_features.append([glyph_train[0], glyph_val[0], glyph_test[0]])
        all_granularity_labels.append([glyph_train[1], glyph_val[1], glyph_test[1]])
        variance_ratios.append(args.variance_ratio_glyph)
        granularity_str += 'glyph, '
        args.log_file.write('glyph ')
        args.log_file.flush()
    granularity_str += 'features.'
    print(granularity_str)

    if args.PREPCA:
        print('use PREPCA!')
        args.log_file.write('use PREPCA!')
    elif args.POSTPCA:
        print('use POSTPCA!')
        args.log_file.write('use POSTPCA!')
    else:
        print('do not use PCA!')

    if len(all_granularity_labels) > 1:
        for data in zip(*all_granularity_labels):
            for d in data[1:]:
                assert (data[0].reshape(-1, 1) == d.reshape(-1, 1)).sum() == len(data[0])

    train_labels = torch.from_numpy(all_granularity_labels[0][0]).float()
    val_labels = torch.from_numpy(all_granularity_labels[0][1]).float()
    test_labels = torch.from_numpy(all_granularity_labels[0][2]).float()

    if args.POSTPCA:
        train_features, val_features, test_features = list(zip(all_granularity_features))
        train_features = np.concatenate(train_features, axis=1)
        val_features = np.concatenate(val_features, axis=1)
        test_features = np.concatenate(test_features, axis=1)
        transform_features = np.concatenate([train_features, val_features, test_features], axis=0)
        W, mu = compute_kernel_bias(transform_features)

        train_features = transform_and_normalize(train_features, W, mu)
        val_features = transform_and_normalize(val_features, W, mu)
        test_features = transform_and_normalize(test_features, W, mu)

    else:
        for i, (train_features, val_features, test_features) in enumerate(all_granularity_features):

            if args.PREPCA:
                print(variance_ratios[i])
                W, mu = compute_kernel_bias(train_features, variance_ratio=variance_ratios[i])
                all_granularity_features[i][0] = transform_and_normalize(train_features, W, mu)
                all_granularity_features[i][1] = transform_and_normalize(val_features, W, mu)
                all_granularity_features[i][2] = transform_and_normalize(test_features, W, mu)
            else:
                if args.normalize:
                    all_granularity_features[i][0] = normalize(train_features)
                    all_granularity_features[i][1] = normalize(val_features)
                    all_granularity_features[i][2] = normalize(test_features)

        train_features, val_features, test_features = list(zip(*all_granularity_features))
        train_features = np.concatenate(train_features, axis=1)
        val_features = np.concatenate(val_features, axis=1)
        test_features = np.concatenate(test_features, axis=1)
    args.log_file.write('\n')
    args.log_file.flush()
    all_data_train = torch.cat([torch.from_numpy(train_features).float(), train_labels.reshape(-1, 1)], dim=1)
    all_data_test = torch.cat([torch.from_numpy(test_features).float(), test_labels.reshape(-1, 1)], dim=1)
    all_data_val = torch.cat([torch.from_numpy(val_features).float(), val_labels.reshape(-1, 1)], dim=1)
    print(train_features.shape)

    print('Number of train: {}, validation: {}, test: {}.'.format(
        len(all_data_train),
        len(all_data_val),
        len(all_data_test)
    ))

    net = nn.Linear(train_features.shape[1], args.num_classes).cuda()

    optimizerSGD = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    loss_function = torch.nn.CrossEntropyLoss()

    if args.sampler:
        print('use class balanced sampler.')
        weights = calculate_sample_weight(train_labels)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
        data_loader_train = DataLoader(all_data_train, sampler=sampler, shuffle=False, batch_size=args.batch_size)
    else:
        data_loader_train = DataLoader(all_data_train, batch_size=args.batch_size, shuffle=False)
    data_loader_test = DataLoader(all_data_test, batch_size=args.batch_size*5, shuffle=False)
    data_loader_val = DataLoader(all_data_val, batch_size=args.batch_size*5, shuffle=False)

    epochs = args.epoch_num
    best_acc_val = 0
    i = 0
    start_time = time.time()
    for e in range(epochs):
        if e % 20 == 0:
            lam = args.lam * i
            i += 1
        loss_mean = train_one_epoch(net, data_loader_train, optimizerSGD, loss_function, lam=lam)
        num = (torch.abs(net.weight) > args.th).sum().cpu().detach().numpy()
        acc_val, sv = test(net, data_loader_val)
        acc_test, st = test(net, data_loader_test)
        if acc_test > best_acc_val:
            best_acc_val = acc_test
            param = net.state_dict()
            torch.save(param, args.path + '/ckpts' + '/{}({}).pth'.format(
        'POSTPCA' if args.POSTPCA else '{}'.format('PREPCA' if args.PREPCA else 'cat'),
                args.feature_name
    ))
        log_str = 'Epoch: [{:3d}/{}]. Loss: {:.4f}. lambda: {}. number: ({}, {}) \n {} (val). {} (test)'.format(
            e + 1,
            epochs,
            loss_mean,
            lam,
            net.weight.shape[1],
            num // args.num_classes,
            sv,
            st
        )
        print(log_str)
        args.log_file.write(log_str + '\n')
        args.log_file.flush()
    end_time = time.time()
    time_cost = end_time - start_time
    args.log_file.write("kept dimension: {}.".format(train_features.shape[1]) + '\n')
    args.log_file.write("time cost: {}.".format(time_cost) + '\n')
    net.load_state_dict(torch.load(args.path + '/ckpts' + '/{}({}).pth'.format(
        'POSTPCA' if args.POSTPCA else '{}'.format('PREPCA' if args.PREPCA else 'cat'),
        args.feature_name
    )))
    _, st = test(net, data_loader_test)
    _, sv = test(net, data_loader_val)
    print(sv + "(val).")
    print(st + "(test).")
    args.log_file.write('\n' + sv + "(val)." + '\n')
    args.log_file.write('\n' + st + "(test)." + '\n')
    args.log_file.flush()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TextClassification')
    parser.add_argument('--gpu', type=str, default='5', help="device id to run")
    parser.add_argument('--epoch_num', type=int, default=200, help="number of epochs")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--lr', type=float, default=5e-2, help="learning rate")
    parser.add_argument('--lam', type=float, default=1e-2, help="trade-off factor")
    parser.add_argument('--th', type=float, default=1e-4, help="threshold")
    parser.add_argument('--variance_ratio_char', type=float, default=0.99, help="tau for char.")
    parser.add_argument('--variance_ratio_word', type=float, default=0.99, help="tau for word")
    parser.add_argument('--variance_ratio_glyph', type=float, default=0.99, help="tau for glyph.")
    parser.add_argument('--variance_ratio_pinyin', type=float, default=0.99, help="tau for pinyin.")
    parser.add_argument('--batch_size', type=float, default=1024, help="batch size")
    parser.add_argument('--optimizer', type=str, default='SGD', help="optimizer")

    parser.add_argument('--PREPCA', action='store_true', help="PCA before concatenation")
    parser.add_argument('--POSTPCA', action='store_true', help="PCA after concatenation")
    parser.add_argument('--sampler', action='store_true', help="for class-unbalanced datasets")
    parser.add_argument('--normalize', action='store_true', help="normalize representation")

    parser.add_argument('--word', type=str, default=None, help="use word-level representation")
    parser.add_argument('--char', type=str, default=None, help="use character-level representation")
    parser.add_argument('--pinyin', type=str, default=None, help="use pinyin-level representation")
    parser.add_argument('--glyph', type=str, default=None, help="use glyph-level representation")
    parser.add_argument('--corpus', type=str, default='wx', help="wx")
    parser.add_argument('--dset', type=str, default='CNT', choices=['TNT', 'CNT', 'FCT'])
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.dset == 'TNT':
        args.num_classes = 15
    elif args.dset == 'CNT':
        args.num_classes = 32
    elif args.dset == 'FCT':
        args.num_classes = 20
    else:
        raise ValueError('Invalid dataset: {}.'.format(args.dset))

    def print_args(args):
        s = "==========================================\n"
        for arg, content in args.__dict__.items():
            s += "{}:{}\n".format(arg, content)
        return s

    args.path = '../intra-granularity-file/{}'.format(args.dset)
    for folder in ['record', 'ckpts']:
        if not os.path.exists(args.path + '/{}'.format(folder)):
            os.system('mkdir -p ' + args.path + '/{}'.format(folder))
        if not os.path.exists(args.path + '/{}'.format(folder)):
            os.mkdir(args.path + '/{}'.format(folder))

    args.feature_name = '{}{}{}{}'.format(
        'c:{},'.format(args.char) if args.char is not None else '',
        'w:{},'.format(args.word) if args.word is not None else '',
        'p:{},'.format(args.pinyin) if args.pinyin is not None else '',
        'g:{},'.format(args.glyph) if args.glyph is not None else ''
    )
    args.log_file = open(args.path + '/record' + '/{}_vr:{}_lam:{}_{}({}).txt'.format(
        args.seed,
        args.variance_ratio_char,
        args.lam,
        'POSTPCA' if args.POSTPCA else '{}'.format('PREPCA' if args.PREPCA else 'cat'),
        args.feature_name
    ), 'w')
    args.log_file.write(print_args(args) + '\n')
    args.log_file.flush()
    train(args)
