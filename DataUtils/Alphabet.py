# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:54
# @File : Alphabet.py
# @Last Modify Time : 2018/1/30 15:54
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Alphabet.py
    FUNCTION : None
"""

import os
import sys
import torch
import random
import collections
from DataUtils.Common import seed_num, unkkey, paddingkey
torch.manual_seed(seed_num)
random.seed(seed_num)


class CreateAlphabet:
    """
        Class:      Create_Alphabet
        Function:   Build Alphabet By Alphabet Class
        Notice:     The Class Need To Change So That Complete All Kinds Of Tasks
    """
    def __init__(self, min_freq=1, train_data=None, dev_data=None, test_data=None, config=None):

        # minimum vocab size
        self.min_freq = min_freq
        self.config = config
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        # storage word and label
        self.word_state = collections.OrderedDict()
        self.label_state = collections.OrderedDict()

        # unk and pad
        self.word_state[unkkey] = self.min_freq
        self.word_state[paddingkey] = self.min_freq

        # word and label Alphabet
        self.word_alphabet = Alphabet(min_freq=self.min_freq)
        self.label_alphabet = Alphabet()

        # unk key
        self.word_unkId = 0
        self.label_unkId = 0

        # padding key
        self.word_paddingId = 0
        self.label_paddingId = 0

    @staticmethod
    def _build_data(train_data=None, dev_data=None, test_data=None):
        """
        :param train_data:
        :param dev_data:
        :param test_data:
        :return:
        """
        assert train_data is not None, "The Train Data Is Not Allow Empty."
        datasets = []
        datasets.extend(train_data)
        print("the length of train data {}".format(len(datasets)))
        if dev_data is not None:
            print("the length of dev data {}".format(len(dev_data)))
            datasets.extend(dev_data)
        if test_data is not None:
            print("the length of test data {}".format(len(test_data)))
            datasets.extend(test_data)
        print("the length of data that create Alphabet {}".format(len(datasets)))
        return datasets

    def build_vocab(self):
        """
        :param train_data:
        :param dev_data:
        :param test_data:
        :param debug_index:
        :return:
        """
        train_data = self.train_data
        dev_data = self.dev_data
        test_data = self.test_data
        print("Build Vocab Start...... ")
        datasets = self._build_data(train_data=train_data, dev_data=dev_data, test_data=test_data)
        # create the word Alphabet

        for index, data in enumerate(datasets):
            # word
            for word in data.words:
                if word not in self.word_state:
                    self.word_state[word] = 1
                else:
                    self.word_state[word] += 1

            # label
            for label in data.labels:
                if label not in self.label_state:
                    self.label_state[label] = 1
                else:
                    self.label_state[label] += 1

        # Create id2words and words2id by the Alphabet Class
        self.word_alphabet.initial(self.word_state)
        self.label_alphabet.initial(self.label_state)

        # unkId and paddingId
        self.word_unkId = self.word_alphabet.from_string(unkkey)
        self.word_paddingId = self.word_alphabet.from_string(paddingkey)

        # fix the vocab
        self.word_alphabet.set_fixed_flag(True)
        self.label_alphabet.set_fixed_flag(True)


class Alphabet:
    """
        Class: Alphabet
        Function: Build vocab
        Params:
              ******    id2words:   type(list),
              ******    word2id:    type(dict)
              ******    vocab_size: vocab size
              ******    min_freq:   vocab minimum freq
              ******    fixed_vocab: fix the vocab after build vocab
              ******    max_cap: max vocab size
    """
    def __init__(self, min_freq=1):
        self.id2words = []
        self.words2id = collections.OrderedDict()
        self.vocab_size = 0
        self.min_freq = min_freq
        self.max_cap = 1e8
        self.fixed_vocab = False

    def initial(self, data):
        """
        :param data:
        :return:
        """
        for key in data:
            if data[key] >= self.min_freq:
                self.from_string(key)
        self.set_fixed_flag(True)

    def set_fixed_flag(self, bfixed):
        """
        :param bfixed:
        :return:
        """
        self.fixed_vocab = bfixed
        if (not self.fixed_vocab) and (self.vocab_size >= self.max_cap):
            self.fixed_vocab = True

    def from_string(self, string):
        """
        :param string:
        :return:
        """
        if string in self.words2id:
            return self.words2id[string]
        else:
            if not self.fixed_vocab:
                newid = self.vocab_size
                self.id2words.append(string)
                self.words2id[string] = newid
                self.vocab_size += 1
                if self.vocab_size >= self.max_cap:
                    self.fixed_vocab = True
                return newid
            else:
                return -1

    def from_id(self, qid, defineStr=""):
        """
        :param qid:
        :param defineStr:
        :return:
        """
        if int(qid) < 0 or self.vocab_size <= qid:
            return defineStr
        else:
            return self.id2words[qid]

    def initial_from_pretrain(self, pretrain_file, unk, padding):
        """
        :param pretrain_file:
        :param unk:
        :param padding:
        :return:
        """
        print("initial alphabet from {}".format(pretrain_file))
        self.initial(unk)
        self.initial(padding)
        now_line = 0
        with open(pretrain_file, encoding="UTF-8") as f:
            for line in f.readlines():
                now_line += 1
                sys.stdout.write("\rhandling with {} line".format(now_line))
                info = line.split(" ")
                self.from_string(info[0])
        f.close()
        print("\nHandle Finished.")



