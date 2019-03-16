# @Author : bamtercelboo
# @Datetime : 2018/10/15 9:52
# @File : CNN.py
# @Last Modify Time : 2018/10/15 9:52
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  CNN.py
    FUNCTION : None
"""

import torch.nn.functional as F
import random
from DataUtils.Common import *
from models.Text_Classification.initialize import *

torch.manual_seed(seed_num)
random.seed(seed_num)


class CNN(nn.Module):
    """
        BiLSTM
    """

    def __init__(self, **kwargs):
        super(CNN, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        Ci = 1
        kernel_nums = self.conv_filter_nums
        kernel_sizes = self.conv_filter_sizes
        paddingId = self.paddingId

        self.embed = nn.Embedding(V, D, padding_idx=paddingId)

        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)
        else:
            init_embedding(self.embed.weight)

        self.dropout_embed = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout)

        # cnn
        if self.wide_conv:
            print("Using Wide Convolution")
            self.conv = [nn.Conv2d(in_channels=Ci, out_channels=kernel_nums, kernel_size=(K, D), stride=(1, 1),
                                   padding=(K // 2, 0), bias=False) for K in kernel_sizes]
        else:
            print("Using Narrow Convolution")
            self.conv = [nn.Conv2d(in_channels=Ci, out_channels=kernel_nums, kernel_size=(K, D), bias=True) for K in kernel_sizes]

        for conv in self.conv:
            if self.device != cpu_device:
                conv.to(self.device)

        in_fea = len(kernel_sizes) * kernel_nums
        self.linear = nn.Linear(in_features=in_fea, out_features=C, bias=True)
        init_linear(self.linear)

    def forward(self, word, sentence_length):
        """
        :param word:
        :param sentence_length:
        :return:
        """
        x = self.embed(word)  # (N,W,D)
        x = self.dropout_embed(x)
        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.linear(x)
        return logit