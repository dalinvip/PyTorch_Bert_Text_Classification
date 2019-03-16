# @Author : bamtercelboo
# @Datetime : 2019/3/16 15:07
# @File : Bert_Encoder.py
# @Last Modify Time : 2019/3/16 15:07
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Bert_Encoder.py
    FUNCTION : None
"""

import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from DataUtils.Common import *
from models.Text_Classification.initialize import *
from models.Text_Classification.modelHelp import prepare_pack_padded_sequence
torch.manual_seed(seed_num)
random.seed(seed_num)


class Bert_Encoder(nn.Module):
    """
        Bert_Encoder
    """

    def __init__(self, **kwargs):
        super(Bert_Encoder, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        self.dropout_bert = nn.Dropout(self.dropout)
        self.bert_linear = nn.Linear(in_features=self.bert_dim, out_features=self.out_dim,
                                     bias=True)
        init_linear_weight_bias(self.bert_linear)

    def forward(self, bert_fea):
        """
        :param bert_fea:
        :return:
        """
        bert_fea = bert_fea.to(self.device)
        bert_fea = bert_fea.permute(0, 2, 1)
        bert_fea = F.max_pool1d(bert_fea, bert_fea.size(2)).squeeze(2)
        bert_fea = self.bert_linear(bert_fea)
        return bert_fea

