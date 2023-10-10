import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel, DistilBertModel, DistilBertConfig
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
from ChineseBERT.modeling_glycebert import GlyceBertForMaskedLM, GlyceBertModel
from datasets.bert_dataset import BertDataset


class NetClassification(nn.Module):
    def __init__(self, embed_dim, embed_data_word, embed_data_char, hidden_size, batch_first, bidirectional, num_classes, use_gpu=None, path=None, args=None):
        super(NetClassification, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        if args.word_func != 'None' and args.word_func == 'lstm':
            self.embedding = nn.Embedding(num_embeddings=len(embed_data_word) + 2, embedding_dim=embed_dim)
            self.embedding.weight.data[2:, :] = embed_data_word
            self.lstm_word = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=batch_first, bidirectional=bidirectional)
            self.ln_embed_word = nn.LayerNorm(embed_dim)
            self.ln_lstm_word = nn.LayerNorm(hidden_size)
            self.drop_embedding_word = nn.Dropout(p=0.5)
            nn.init.orthogonal_(self.lstm_word.weight_ih_l0)
            nn.init.orthogonal_(self.lstm_word.weight_hh_l0)
            self.lstm_word.bias_ih_l0 = torch.nn.parameter.Parameter(torch.zeros_like(self.lstm_word.bias_ih_l0))
            self.lstm_word.bias_hh_l0 = torch.nn.parameter.Parameter(torch.zeros_like(self.lstm_word.bias_hh_l0))

        CHINESEBERT_PATH = '../ckpts/ChineseBERT-base'
        self.tokenizer_CHN = None

        if args.pinyin_func != 'None' or args.glyph_func != 'None' or args.char_func == 'lstm':
            self.tokenizer_CHN = BertDataset(CHINESEBERT_PATH)
            chinese_bert = GlyceBertModel.from_pretrained(CHINESEBERT_PATH)

        if args.char_func != 'None':
            if args.char_func == 'bert':
                if args.bert_name[:len('ChineseBERT')] == 'ChineseBERT':
                    if self.tokenizer_CHN is None:
                        self.tokenizer_CHN = BertDataset(CHINESEBERT_PATH)
                    self.chinese_bert = GlyceBertModel.from_pretrained(CHINESEBERT_PATH)
                else:
                    self.bert = AutoModel.from_pretrained(args.bert_name)
                    self.bertTokenizer = AutoTokenizer.from_pretrained(args.bert_name)
                self.ln_bert = nn.LayerNorm(hidden_size)
                self.ln_bert_avg = nn.LayerNorm(hidden_size)

            if args.char_func == 'lstm':
                self.embedding_char = nn.Embedding(num_embeddings=len(embed_data_char) + 2, embedding_dim=embed_dim)
                self.char_embeddings = chinese_bert.embeddings.word_embeddings
                self.embedding_char.weight.data[2:, :] = embed_data_char
                self.lstm_char = nn.LSTM(input_size=768, hidden_size=hidden_size, batch_first=batch_first,
                                         bidirectional=bidirectional)
                self.ln_embed_char = nn.LayerNorm(768)
                self.ln_lstm_char = nn.LayerNorm(hidden_size)
                self.drop_embedding_char = nn.Dropout(p=0.5)
                nn.init.orthogonal_(self.lstm_char.weight_ih_l0)
                nn.init.orthogonal_(self.lstm_char.weight_hh_l0)
                self.lstm_char.bias_ih_l0 = torch.nn.parameter.Parameter(torch.zeros_like(self.lstm_char.bias_ih_l0))
                self.lstm_char.bias_hh_l0 = torch.nn.parameter.Parameter(torch.zeros_like(self.lstm_char.bias_hh_l0))

        if args.pinyin_func != 'None':
            self.lstm_pinyin = nn.LSTM(input_size=768, hidden_size=hidden_size, batch_first=batch_first,
                                     bidirectional=bidirectional)
            self.pinyin_embeddings = chinese_bert.embeddings.pinyin_embeddings
            self.ln_embed_pinyin = nn.LayerNorm(768)
            self.ln_lstm_pinyin = nn.LayerNorm(hidden_size)
            self.drop_embedding_pinyin = nn.Dropout(p=0.5)
            nn.init.orthogonal_(self.lstm_pinyin.weight_ih_l0)
            nn.init.orthogonal_(self.lstm_pinyin.weight_hh_l0)
            self.lstm_pinyin.bias_ih_l0 = torch.nn.parameter.Parameter(torch.zeros_like(self.lstm_pinyin.bias_ih_l0))
            self.lstm_pinyin.bias_hh_l0 = torch.nn.parameter.Parameter(torch.zeros_like(self.lstm_pinyin.bias_hh_l0))

        if args.glyph_func != 'None':
            self.lstm_glyph = nn.LSTM(input_size=768, hidden_size=hidden_size, batch_first=batch_first,
                                       bidirectional=bidirectional)
            self.glyph_embeddings = nn.Sequential(
                chinese_bert.embeddings.glyph_embeddings,
                chinese_bert.embeddings.glyph_map
            )
            self.ln_embed_glyph = nn.LayerNorm(768)
            self.ln_lstm_glyph = nn.LayerNorm(hidden_size)
            self.drop_embedding_glyph = nn.Dropout(p=0.5)
            nn.init.orthogonal_(self.lstm_glyph.weight_ih_l0)
            nn.init.orthogonal_(self.lstm_glyph.weight_hh_l0)
            self.lstm_glyph.bias_ih_l0 = torch.nn.parameter.Parameter(torch.zeros_like(self.lstm_glyph.bias_ih_l0))
            self.lstm_glyph.bias_hh_l0 = torch.nn.parameter.Parameter(torch.zeros_like(self.lstm_glyph.bias_hh_l0))

        num_granularity = sum([
            args.pinyin_func != 'None',
            args.glyph_func != 'None',
            args.char_func != 'None',
            args.word_func != 'None'
        ])
        print('Number of granularity features: {}'.format(num_granularity))
        self.head = nn.Linear(hidden_size*num_granularity, num_classes)
        nn.init.xavier_normal_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.criterion = nn.CrossEntropyLoss()
        self.use_gpu = use_gpu
        self.drop_fc = nn.Dropout(p=0.5)
        self.device = 'cuda:{}'.format(args.gpu)

        if path is not None:
            self.load_state_dict(torch.load(path))

    def forward(self, inputs):
        y = torch.Tensor(inputs[0]).long().to(self.device)
        word_embed = None
        char_embed = None
        pinyin_embed = None
        glyph_embed = None

        if self.args.word_func != 'None' and self.args.word_func == 'lstm':
            word_embed = self.get_word_features(inputs)

        if self.args.char_func != 'None':
            if self.args.char_func == 'bert':
                if self.args.bert_name[:len('ChineseBERT')] == 'ChineseBERT':
                    bert_output, cls_embed, attn_mask = self.chinese_bert_forward(inputs)
                else:
                    bert_output, cls_embed, attn_mask = self.bert_forward(inputs)
                if self.args.use_bert_avg:
                    char_embed = self.ln_bert_avg(
                        (bert_output * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(1).reshape(-1, 1))
                else:
                    char_embed = self.ln_bert(cls_embed)

            if self.args.char_func == 'lstm':
                char_embed = self.get_char_features(inputs)

        if self.args.pinyin_func != 'None':
            pinyin_embed = self.get_pinyin_features(inputs)

        if self.args.glyph_func != 'None':
            glyph_embed = self.get_glyph_features(inputs)

        features = [word_embed, char_embed, pinyin_embed, glyph_embed]
        sentence_embed = []
        for feature in features:
            if feature is not None:
                sentence_embed.append(feature.reshape(-1, self.hidden_size))

        if len(sentence_embed) == 1:
            sentence_embed = sentence_embed[0]
        else:
            sentence_embed = torch.cat(sentence_embed, dim=1)

        logits = self.head(self.drop_fc(sentence_embed)).squeeze()

        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)

        if self.training:
            loss = self.criterion(logits, y)
            return loss
        else:
            _, pred = logits.max(1)
            tf = pred.float().eq(y.float())
            return pred, tf.sum().detach().cpu().numpy(), len(tf), sentence_embed, y

    def evaluate(self, test_loader, visualize_feature=False, vanilla=''):
        print('starting evaluation!')
        self.eval()
        correct_num, total_num = 0, 0
        start = True
        pred_list = []
        label_list = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                pred, correct, total, features, y = self.forward(data)
                correct_num += correct
                total_num += total
                pred_list.extend(list(pred.detach().cpu().numpy().astype('int64')))
                label_list.extend(list(y.detach().cpu().numpy().astype('int64')))
                if visualize_feature:
                    if start:
                        all_feature = features.cpu().detach().float()
                        all_label = y.cpu().detach().float()
                        start = False
                    else:
                        all_feature = torch.cat([all_feature, features.cpu().detach().float()], dim=0)
                        all_label = torch.cat([all_label, y.cpu().detach().float()], dim=0)
        assert len(label_list) == len(pred_list)
        acc = correct_num / (total_num + 1e-8)
        if self.args.num_classes == 2:
            print('binary classification task.')
            f1_macro = f1_score(label_list, pred_list)
            recall_macro = recall_score(label_list, pred_list)
            precision_macro = precision_score(label_list, pred_list)
        else:
            f1_macro = f1_score(label_list, pred_list, labels=[i for i in range(self.num_classes)], average="macro")
            recall_macro = recall_score(label_list, pred_list, labels=[i for i in range(self.num_classes)], average="macro")
            precision_macro = precision_score(label_list, pred_list, labels=[i for i in range(self.num_classes)],
                                              average="macro")

        log_str = 'Accuracy={:.4f}. Precision={:.4f}. Recall={:.4f}. F1={:.4f}'.format(
            acc,
            precision_macro,
            recall_macro,
            f1_macro
        )

        if visualize_feature:
            all_feature_numpy = all_feature.numpy()
            all_label_numpy = all_label.numpy().astype('int64')
            save_path = '../log/{}/features/{}_{}_char:{}_word:{}_pinyin:{}_glyph:{}_avg:{}_'.format(
                self.args.dset, self.args.seed, self.args.corpus, self.args.bert_name.split('/')[-1] if self.args.char_func == 'bert' else self.args.char_func, self.args.word_func,
                self.args.pinyin_func,
                self.args.glyph_func,
                str(self.args.use_bert_avg)
            )
            np.save(save_path+vanilla+'_features.npy', all_feature_numpy)
            np.save(save_path+vanilla+'_labels.npy', all_label_numpy)
        self.train()
        return f1_macro, log_str

    def predict(self, inputs):
        self.eval()
        pred, _, _, _, _ = self.forward(inputs)
        return pred

    def get_char_features(self, inputs):
        if self.args.char_func == 'lstm':
            return self.lstm_char_forward(inputs)
        elif self.args.char_func == 'bert':
            return self.bert_forward(inputs)

    def get_word_features(self, inputs):
        return self.lstm_word_forward(inputs)

    def get_pinyin_features(self, inputs):
        return self.lstm_pinyin_forward(inputs)

    def get_glyph_features(self, inputs):
        return self.lstm_glyph_forward(inputs)

    def lstm_forward(self, inputs):
        y, x, raw_txt = inputs
        idx = torch.LongTensor([i for i in range(len(y))]).unsqueeze(1)
        max_len = max([len(s) for s in x])
        x = torch.Tensor([s + ([-2] * (max_len - len(s))) for s in x]).long().to(self.device)
        x = self.drop_embedding(self.embedding(x + 2))
        mask = torch.Tensor([len(s) - 1 for s in x]).long().unsqueeze(1)
        outputs, (hn, cn) = self.lstm(self.ln_embed(x))
        sentence_embed = self.ln_lstm(outputs[idx, mask].squeeze())
        return sentence_embed

    def bert_forward(self, inputs):
        y, _, _,raw_txt = inputs
        max_len = min(max([len(s) + 2 for s in raw_txt]), self.args.max_len)
        inputs_for_bert = self.bertTokenizer(list(raw_txt), padding=True,
                                             truncation=True,
                                             max_length=max_len,
                                             return_tensors="pt").to(self.device)
        bert_output, cls_embed = self.bert(**inputs_for_bert)
        return bert_output, cls_embed, inputs_for_bert['attention_mask']

    def chinese_bert_forward(self, inputs):
        y, _, _, raw_txt = inputs
        max_len = min(max([len(s) + 2 for s in raw_txt]), self.args.max_len)
        batch_inputs_ids = torch.zeros(len(y), max_len, device=self.device, dtype=torch.float32)
        batch_pinyin_ids = torch.zeros(len(y), max_len, 8, device=self.device, dtype=torch.float32)
        for i, sen in enumerate(raw_txt):
            input_ids, pinyin_ids = self.tokenizer_CHN.tokenize_sentence(sen[:max_len-2])
            length = input_ids.shape[0]
            input_ids = input_ids.view(1, length)
            pinyin_ids = pinyin_ids.view(1, length, 8)
            batch_inputs_ids[i, :length] = input_ids.to(self.device)
            batch_pinyin_ids[i, :length, :] = pinyin_ids.to(self.device)

        attention_mask = (batch_inputs_ids != 0).long()
        output_hidden = self.chinese_bert(batch_inputs_ids.long(), batch_pinyin_ids.long(), attention_mask)
        return output_hidden[0], output_hidden[1], attention_mask

    def lstm_word_forward(self, inputs):
        y, x, _, raw_txt = inputs

        idx = torch.tensor([i for i in range(len(y))], device=self.device).unsqueeze(1).long()
        max_len = min(max([len(s) for s in x]), self.args.max_len)
        mask = torch.tensor([min(len(s)-1, max_len-1) for s in x], device=self.device).long().unsqueeze(1)

        x = torch.Tensor([s[:self.args.max_len] + ([-2] * (max_len - len(s))) for s in x]).long().to(self.device)
        x = self.drop_embedding_word(self.embedding(x + 2))
        outputs, (hn, cn) = self.lstm_word(self.ln_embed_word(x))
        sentence_embed = self.ln_lstm_word(outputs[idx, mask].squeeze())
        return sentence_embed

    def lstm_char_forward(self, inputs):
        y, _, _, raw_txt = inputs
        idx = torch.tensor([i for i in range(len(y))], device=self.device).unsqueeze(1).long()
        max_len = min(max([len(s) + 2 for s in raw_txt]), self.args.max_len)
        mask = torch.tensor([min(len(s)-1, max_len-1) for s in raw_txt], device=self.device).long().unsqueeze(1)
        batch_inputs_ids = torch.zeros(len(y), max_len, device=self.device, dtype=torch.float32)
        for i, sen in enumerate(raw_txt):
            input_ids, pinyin_ids = self.tokenizer_CHN.tokenize_sentence(sen[:max_len-2])
            length = input_ids.shape[0]
            input_ids = input_ids.view(1, length)
            batch_inputs_ids[i, :length] = input_ids.to(self.device).float()
        char_embeddings = self.drop_embedding_char(self.char_embeddings(batch_inputs_ids.long()))
        outputs, (hn, cn) = self.lstm_char(self.ln_embed_char(char_embeddings))
        sentence_embed = self.ln_lstm_char(outputs[idx, mask].squeeze())

        return sentence_embed

    def lstm_pinyin_forward(self, inputs):
        y, _, _, raw_txt = inputs
        idx = torch.tensor([i for i in range(len(y))], device=self.device).unsqueeze(1).long()
        max_len = min(max([len(s) + 2 for s in raw_txt]), self.args.max_len)
        mask = torch.tensor([min(len(s) - 1, max_len - 1) for s in raw_txt], device=self.device).long().unsqueeze(1)
        batch_pinyin_ids = torch.zeros(len(y), max_len, 8, device=self.device, dtype=torch.float32)
        for i, sen in enumerate(raw_txt):
            input_ids, pinyin_ids = self.tokenizer_CHN.tokenize_sentence(sen[:max_len-2])
            length = input_ids.shape[0]
            pinyin_ids = pinyin_ids.view(1, length, 8)
            batch_pinyin_ids[i, :length, :] = pinyin_ids.to(self.device)
        pinyin_embeddings = self.drop_embedding_pinyin(self.pinyin_embeddings(batch_pinyin_ids.long()))
        outputs, (hn, cn) = self.lstm_pinyin(self.ln_embed_pinyin(pinyin_embeddings))
        sentence_embed = self.ln_lstm_pinyin(outputs[idx, mask].squeeze())

        return sentence_embed

    def lstm_glyph_forward(self, inputs):
        y, _, _, raw_txt = inputs
        idx = torch.tensor([i for i in range(len(y))], device=self.device).unsqueeze(1).long()
        max_len = min(max([len(s) + 2 for s in raw_txt]), self.args.max_len)
        mask = torch.tensor([min(len(s) - 1, max_len - 1) for s in raw_txt], device=self.device).long().unsqueeze(1)
        batch_inputs_ids = torch.zeros(len(y), max_len, device=self.device, dtype=torch.float32)
        for i, sen in enumerate(raw_txt):
            input_ids, pinyin_ids = self.tokenizer_CHN.tokenize_sentence(sen[:max_len-2])
            length = input_ids.shape[0]
            input_ids = input_ids.view(1, length)
            batch_inputs_ids[i, :length] = input_ids.to(self.device)
        glyph_embeddings = self.drop_embedding_glyph(self.glyph_embeddings(batch_inputs_ids.long()))
        outputs, (hn, cn) = self.lstm_glyph(self.ln_embed_glyph(glyph_embeddings))
        sentence_embed = self.ln_lstm_glyph(outputs[idx, mask].squeeze())
        return sentence_embed