import gensim
import torch
import os
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
from data_provider import data_load, data_preprocess, FCT_preprocess
from transformers import AdamW
from model import NetClassification
import numpy as np
import random
import warnings
from collections import Counter
warnings.filterwarnings('ignore')


class AvgLoss:
    def __init__(self):
        self.loss = 0.0
        self.n = 0

    def add_loss(self, loss):
        self.loss += loss
        self.n += 1

    def get_avg_loss(self):
        loss = self.loss / self.n
        return loss

    def clear(self):
        self.loss = 0.0
        self.n = 0


def calculate_sample_weight(dataset):
    labels = [int(d[0]) for d in dataset]
    freq = Counter(labels)
    weights = torch.Tensor([freq[l] for l in labels])
    return 1. / weights


def train(args):
    corpus = args.corpus
    embed_data_word = None
    embed_data_char = None
    if args.word_func != 'None':
        print('using {} corpus to initialize the word embedding matrix.'.format(corpus))
        model_word = gensim.models.Word2Vec.load('../file/word2vec_{}'.format(corpus))
        embed_data_word = torch.from_numpy(model_word.wv.vectors)

    if args.char_func != 'None':
        print('using {} corpus to initialize the char embedding matrix.'.format(corpus))
        model_char = gensim.models.Word2Vec.load('../file/word2vec_wiki')
        embed_data_char = torch.from_numpy(model_char.wv.vectors)

    def collate_fn(data):
        return tuple(zip(*data))

    print('preparing data!')
    if args.dset == 'TNT':
        train_list = data_load(args.corpus, 'train')
        val_list = data_load(args.corpus, 'val')
        test_list = data_load(args.corpus, 'test')
    elif args.dset == 'CNT':
        test_list = data_preprocess(args.corpus, 'test', args.val_ratio)
        train_list, val_list = data_preprocess(args.corpus, 'train', args.val_ratio)
    elif args.dset == 'FCT':
        test_list = FCT_preprocess(args.corpus, 'test', args.val_ratio)
        train_list, val_list = FCT_preprocess(args.corpus, 'train', args.val_ratio)
    else:
        raise ValueError('Invalid dataset: {}.'.format(args.dset))
    print('Len. of training data: {}, val data: {}, test data: {}.'.format(
        len(train_list),
        len(val_list),
        len(test_list)
    ))
    best_acc_val = 0.0
    if args.sampler and args.train:
        print('use class balanced sampler.')
        weights = calculate_sample_weight(train_list)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
        train_loader = DataLoader(train_list, sampler=sampler, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size*(1 + (1 - args.train) * 4))
    else:
        train_loader = DataLoader(train_list, batch_size=args.batch_size*(1 + (1 - args.train) * 4), collate_fn=collate_fn, shuffle=args.train)
    train_loader_no_shuffle = DataLoader(train_list, batch_size=args.batch_size*5, collate_fn=collate_fn, shuffle=False)
    print(args.batch_size*(1 + (1 - args.train) * 4))
    val_loader = DataLoader(val_list, batch_size=args.batch_size*5, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=args.batch_size*5, collate_fn=collate_fn, shuffle=False)
    if args.bert_name[-5:].lower() == 'large':
        hidden_size = 1024
    else:
        hidden_size = 768
    lr = args.lr
    print('creating model!')
    net = NetClassification(embed_dim=256, embed_data_word=embed_data_word, embed_data_char=embed_data_char, hidden_size=hidden_size, batch_first=True,
              bidirectional=False, num_classes=args.num_classes, use_gpu=args.gpu, args=args).cuda(args.gpu)

    if args.train:
        avg_loss = AvgLoss()
        param_list = []
        for k, v in net.named_parameters():
            if k[:4] == 'head':
                param_list.append({'params': v, 'lr': lr})
            elif k[:5] == 'embed':
                if args.lr_decay1 <= 0:
                    v.requires_grad = False
                else:
                    param_list.append({'params': v, 'lr': lr * args.lr_decay1})
            elif k[:4] == 'bert':
                if args.lr_decay2 <= 0:
                    v.requires_grad = False
                else:
                    param_list.append({'params': v, 'lr': lr * args.lr_decay2})
            elif 'embeddings' in k:
                if args.lr_decay3 <= 0:
                    v.requires_grad = False
                else:
                    param_list.append({'params': v, 'lr': lr * args.lr_decay3})
            else:
                param_list.append({'params': v, 'lr': lr})

        if args.optimizer == 'SGD' and any([args.char_func == 'lstm', args.char_func == 'None']):
            print('use SGD optimizer!')
            optimizer = torch.optim.SGD(param_list, momentum=0.9, weight_decay=5e-4, nesterov=True)
        if args.char_func != 'lstm' and args.char_func != 'None':
            print('use AdamW optimizer!')
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 2e-3,
                },
                {
                    "params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=args.lr,
                              eps=1e-8)

        print('starting training!')
        acc_list = []
        loss_list = []
        step_list = []
        iter_num = 0
        for i in range(args.epoch_num):
            net.train()
            for j, d in enumerate(train_loader):
                optimizer.zero_grad()
                loss = net(d)
                loss.backward()
                avg_loss.add_loss(loss.item())
                optimizer.step()
                iter_num += 1
                if iter_num % args.interval == 0:
                    print('Epoch: [{}/{}], Iter: [{}/{}], Loss: {:.4f}'.format(i+1, args.epoch_num, j+1, len(train_loader), avg_loss.get_avg_loss()))

                    acc_val, s_val = net.evaluate(val_loader)
                    acc_list.append(acc_val)
                    loss_list.append(avg_loss.get_avg_loss())
                    step_list.append(iter_num)
                    avg_loss.clear()

                    args.out_file.write('Epoch: [{}/{}]. '.format(i+1, args.epoch_num) + s_val + '\n')
                    args.out_file.flush()
                    print('Epoch: [{}/{}]. '.format(i+1, args.epoch_num) + s_val + '\n')
                    if best_acc_val < acc_val:
                        params_dict = net.state_dict()
                        best_acc_val = acc_val
                        args.out_file.write(str(acc_val) + '\n')
                        args.out_file.flush()
                        save_path = '../log/{}/ckpts/{}_{}_char:{}_word:{}_pinyin:{}_glyph:{}_avg:{}.pth'.format(
                            args.dset, args.seed, args.corpus, args.bert_name.split('/')[-1] if args.char_func == 'bert' else args.char_func, args.word_func,
                            args.pinyin_func,
                            args.glyph_func,
                            str(args.use_bert_avg)
                        )
                        torch.save(params_dict, save_path)
        acc_file = np.array(acc_list).reshape(-1, 1)
        loss_file = np.array(loss_list).reshape(-1, 1)
        step_file = np.array(step_list).reshape(-1, 1)
        np.save(save_path[:-4] + '_acc.npy', np.concatenate([step_file, acc_file], axis=1))
        np.save(save_path[:-4] + '_loss.npy', np.concatenate([step_file, loss_file], axis=1))
        net.load_state_dict(torch.load(save_path))
        net.eval()
        acc_train, s_train = net.evaluate(train_loader_no_shuffle, visualize_feature=True, vanilla='train_set')
        acc_val, s_val = net.evaluate(val_loader, visualize_feature=True, vanilla='val_set')
        acc_test, s_test = net.evaluate(test_loader, visualize_feature=True, vanilla='test_set')
        args.out_file.write('\n' + s_train + " (train)." + '\n')
        args.out_file.write(s_val + " (val)." + '\n')
        args.out_file.write(s_test + " (test)." + '\n')
        args.out_file.flush()
    else:
        print('starting evaluation!')
        save_path = '../log/{}/ckpts/{}_{}_char:{}_word:{}_pinyin:{}_glyph:{}_avg:{}.pth'.format(
            args.dset, args.seed, args.corpus, args.bert_name.split('/')[-1] if args.char_func == 'bert' else args.char_func, args.word_func,
            args.pinyin_func,
            args.glyph_func,
            str(args.use_bert_avg)
        )
        net.load_state_dict(torch.load(save_path))
        acc_train, s_train = net.evaluate(train_loader, visualize_feature=True, vanilla='train_set')
        acc_val, s_val = net.evaluate(val_loader, visualize_feature=True, vanilla='val_set')
        acc_test, s_test = net.evaluate(test_loader, visualize_feature=True, vanilla='test_set')
        args.out_file.write('\n' + s_train + " (train)." + '\n')
        args.out_file.write(s_val + " (val)." + '\n')
        args.out_file.write(s_test + " (test)." + '\n')
        args.out_file.flush()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TextClassification')
    parser.add_argument('--gpu', type=int, default=4, help="device id to run")
    parser.add_argument('--epoch_num', type=int, default=10, help="number of epochs")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_decay1', type=float, default=0.1, help="learning rate decay")
    parser.add_argument('--lr_decay2', type=float, default=0.1, help="learning rate decay")
    parser.add_argument('--lr_decay3', type=float, default=0.0, help="learning rate decay")
    parser.add_argument('--val_ratio', type=float, default=0.1, help="val ratio")
    # parser.add_argument('--beta', type=float, default=0.0, help="learning rate decay")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--hidden_size', type=int, default=768, help="hidden size")
    parser.add_argument('--interval', type=int, default=1000, help="print interval")
    parser.add_argument('--max_len', type=int, default=512, help="batch size")
    parser.add_argument('--optimizer', type=str, default='SGD', help="optimizer")

    parser.add_argument('--char_func', type=str, default='None', help="char embed")
    parser.add_argument('--word_func', type=str, default='None', help="word embed")
    parser.add_argument('--pinyin_func', type=str, default='None', help="word embed")
    parser.add_argument('--glyph_func', type=str, default='None', help="word embed")
    parser.add_argument('--use_bert_avg', action='store_true', help="BERT+LSTM")
    parser.add_argument('--sampler', action='store_true', help="for class-unbalanced datasets")
    parser.add_argument('--train', action='store_false', help="training or test")
    parser.add_argument('--corpus', type=str, default='wx', choices=['wx', 'baike'])
    parser.add_argument('--dset', type=str, default='CNT', choices=['TNT', 'CNT', 'FCT'])
    parser.add_argument('--bert_name', type=str, default='hfl/chinese-bert-wwm', choices=[
        'hfl/chinese-bert-wwm', 'hfl/chinese-roberta-wwm-ext', 'hfl/chinese-bert-wwm-ext', 'ChineseBERT-base',
        'ChineseBERT-large'
    ])
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.enabled = False

    for folder in ['record', 'features', 'ckpts']:
        path = '../log/{}/{}'.format(args.dset, folder)
        if not os.path.exists(path):
            os.system('mkdir -p ' + path)
        if not os.path.exists(path):
            os.mkdir(path)

    args.path = '../lasso/{}'.format(args.dset)
    if not os.path.exists(args.path):
        os.system('mkdir -p ' + args.path)
    if not os.path.exists(args.path):
        os.mkdir(args.path)

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

    record_path = '../log/{}/record/{}_train_{}_char:{}_word:{}_pinyin:{}_glyph:{}_avg:{}.txt'.format(
        args.dset, args.seed, args.corpus, args.bert_name.split('/')[-1] if args.char_func == 'bert' else args.char_func, args.word_func,
        args.pinyin_func,
        args.glyph_func,
        str(args.use_bert_avg)
    )
    args.out_file = open(record_path, 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train(args)