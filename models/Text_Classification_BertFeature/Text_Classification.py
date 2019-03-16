# @Author : bamtercelboo
# @Datetime : 2018/9/14 8:43
# @File : Sequence_Label.py
# @Last Modify Time : 2018/9/14 8:43
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Sequence_Label.py
    FUNCTION : None
"""

import torch
import torch.nn as nn
import random
from models.Text_Classification_BertFeature.BiLSTM import BiLSTM
from models.Text_Classification_BertFeature.Bert_Encoder import Bert_Encoder
from models.Text_Classification.CNN import CNN
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Text_Classification_BertFeature(nn.Module):
    """
        Text_Classification
    """

    def __init__(self, config):
        super(Text_Classification_BertFeature, self).__init__()
        self.config = config
        # embed
        self.embed_num = config.embed_num
        self.embed_dim = config.embed_dim
        self.label_num = config.label_num
        self.paddingId = config.paddingId
        # dropout
        self.dropout_emb = config.dropout_emb
        self.dropout = config.dropout
        # lstm
        self.lstm_hiddens = config.lstm_hiddens
        self.lstm_layers = config.lstm_layers
        # pre train
        self.pretrained_embed = config.pretrained_embed
        self.pretrained_weight = config.pretrained_weight
        # cnn param
        self.wide_conv = config.wide_conv
        self.conv_filter_sizes = self._conv_filter(config.conv_filter_sizes)
        self.conv_filter_nums = config.conv_filter_nums
        # self.use_cuda = config.use_cuda
        self.device = config.device

        self.bert_out_dim = 200

        self.model = BiLSTM(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                            paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                            lstm_hiddens=self.lstm_hiddens, lstm_layers=self.lstm_layers,
                            pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                            device=self.device, bert_out_dim=self.bert_out_dim)

        self.Bert_Encoder = Bert_Encoder(dropout=0.5, bert_dim=config.bert_dim,
                                         out_dim=self.bert_out_dim, device=self.device)

    @staticmethod
    def _conv_filter(str_list):
        """
        :param str_list:
        :return:
        """
        int_list = []
        str_list = str_list.split(",")
        for str in str_list:
            int_list.append(int(str))
        return int_list

    @staticmethod
    def _get_model_args(batch_features):
        """
        :param batch_features:  Batch Instance
        :return:
        """
        word = batch_features.word_features
        mask = word > 0
        sentence_length = batch_features.sentence_length
        labels = batch_features.label_features
        batch_size = batch_features.batch_length
        bert_feature = batch_features.bert_features
        return word, bert_feature, mask, sentence_length, labels, batch_size

    def forward(self, batch_features, train=False):
        """
        :param batch_features:
        :param train:
        :return:
        """
        word, bert_feature, mask, sentence_length, labels, batch_size = self._get_model_args(batch_features)
        bert_fea = self.Bert_Encoder(bert_feature)
        model_output = self.model(word, bert_fea, sentence_length)
        return model_output


