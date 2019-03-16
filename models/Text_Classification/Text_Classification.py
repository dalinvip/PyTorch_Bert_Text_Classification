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
from models.Text_Classification.BiLSTM import BiLSTM
from models.Text_Classification.CNN import CNN
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Text_Classification(nn.Module):
    """
        Text_Classification
    """

    def __init__(self, config):
        super(Text_Classification, self).__init__()
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

        if self.config.model_bilstm:
            self.model = BiLSTM(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                                paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                                lstm_hiddens=self.lstm_hiddens, lstm_layers=self.lstm_layers,
                                pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                                device=self.device)
        elif self.config.model_cnn:
            self.model = CNN(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                             paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                             conv_filter_nums=self.conv_filter_nums, conv_filter_sizes=self.conv_filter_sizes,
                             wide_conv=self.wide_conv,
                             pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                             device=self.device)

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

    def forward(self, word, sentence_length, train=False):
        """
        :param char:
        :param word:
        :param sentence_length:
        :param train:
        :return:
        """

        model_output = self.model(word, sentence_length)
        return model_output


