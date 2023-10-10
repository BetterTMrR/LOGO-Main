# Learning Complementary Granularity Representations for Chinese TextClassification


This is a Pytorch implementation of "Learning Complementary Granularity Representations for Chinese TextClassification".


## Installation

`pip install -r requirements.txt`


## Data Preparation

Please download datasets [CNT](http://pan.baidu.com/s/1mgBTFOO) (this dataset has been provided in ./file/), [FCT](https://www.heywhale.com/mw/dataset/5d3a9c86cf76a600360edd04), [TNT](https://www.heywhale.com/mw/dataset/5dd645fca0cb22002c94e65d/file).

CNT dataset: ./file/train_file.text; ./file/test_file.text;

TNT dataset: ./file/train.tsv; ./file/val.tsv; ./file/test.tsv;

FCT dataset: ./file/FCT;

## Preparation for Word Embedding and Chinese BERT

Please download [Word embedding model](https://spaces.ac.cn/archives/4304) to ./file/word2vec_wx, ./file/word2vec_wx.syn1neg.npy, ./file/word2vec_wx.syn1.npy, ./file/word2vec_wx.wv.syn0.npy.
The pinyin, glyph, and character embedding models are provided by [Chinese BERT](https://arxiv.org/abs/2106.16038), and please download ChineseBERT-base model to ./ckpts.


## Training

cd ./code

(1) obtain character- and word-level representations using LSTM (BERT) model:

`python train.py --char_func lstm --gpu 0 --corpus wx --dset CNT --lr 1e-2 --val_ratio 0.1 --epoch_num 10 --sampler
 `
 
 `python train.py --char_func bert --gpu 0 --corpus wx --dset CNT --lr 1e-5 --val_ratio 0.1 --epoch_num 5 --sampler
 `
 
 `python train.py --word_func lstm --gpu 0 --corpus wx --dset CNT --lr 1e-2 --val_ratio 0.1 --epoch_num 10 --sampler
 `
 

(2) learning intra-granularity independent signals:

`python intra-granularity.py --gpu 0 --dset CNT --PREPCA --word lstm --char lstm (or bert) --sampler
 `
 
 The training log of CNT dataset (with LSTM models) can be found in ./intra-granularity-file/CNT/record
