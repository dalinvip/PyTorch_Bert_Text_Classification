# @Author : bamtercelboo
# @Datetime : 2018/1/30 15:58
# @File : DataConll2003_Loader.py
# @Last Modify Time : 2018/1/30 15:58
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :
    FUNCTION :
"""
import sys
import os
import re
import random
import torch
import json
from Dataloader.Instance import Instance

from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class DataLoaderHelp(object):
    """
    DataLoaderHelp
    """

    @staticmethod
    def _clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def _normalize_word(word):
        """
        :param word:
        :return:
        """
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    @staticmethod
    def _sort(insts):
        """
        :param insts:
        :return:
        """
        sorted_insts = []
        sorted_dict = {}
        for id_inst, inst in enumerate(insts):
            sorted_dict[id_inst] = inst.words_size
        dict = sorted(sorted_dict.items(), key=lambda d: d[1], reverse=True)
        for key, value in dict:
            sorted_insts.append(insts[key])
        print("Sort Finished.")
        return sorted_insts


class DataLoader(DataLoaderHelp):
    """
    DataLoader
    """
    def __init__(self, path, shuffle, config):
        """
        :param path: data path list
        :param shuffle:  shuffle bool
        :param config:  config
        """
        #
        print("Loading Data......")
        self.data_list = []
        self.max_count = config.max_count
        self.path = path
        self.shuffle = shuffle

        # BERT
        self.bert_path = [config.bert_train_file,
                          config.bert_dev_file,
                          config.bert_test_file]

        self.use_bert = config.use_bert

    def dataLoader(self):
        """
        :return:
        """
        path = self.path
        shuffle = self.shuffle
        assert isinstance(path, list), "Path Must Be In List"
        print("Data Path {}".format(path))
        for id_data in range(len(path)):
            print("Loading Data Form {}".format(path[id_data]))
            insts = self._Load_Each_Data(path=path[id_data], path_id=id_data)
            print("shuffle train data......")
            random.shuffle(insts)
            self.data_list.append(insts)
        # return train/dev/test data
        if len(self.data_list) == 3:
            return self.data_list[0], self.data_list[1], self.data_list[2]
        elif len(self.data_list) == 2:
            return self.data_list[0], self.data_list[1]

    def _Load_Each_Data(self, path=None, path_id=None):
        """
        :param path:
        :param shuffle:
        :return:
        """
        assert path is not None, "The Data Path Is Not Allow Empty."
        insts = []
        now_lines = 0
        with open(path, encoding="UTF-8") as f:
            inst = Instance()
            for line in f.readlines():
                line = line.strip()
                now_lines += 1
                if now_lines % 200 == 0:
                    sys.stdout.write("\rreading the {} line\t".format(now_lines))
                if line == "\n":
                    print("empty line")

                inst = Instance()
                line = line.split()
                label = line[0]
                word = " ".join(line[1:])
                if label not in ["0", "1"]:
                    print("Error line: ", " ".join(line))
                    continue
                inst.words = self._clean_str(word).split()
                inst.labels.append(label)
                inst.words_size = len(inst.words)
                insts.append(inst)

                if len(insts) == self.max_count:
                    break
            # print("\n")
        if self.use_bert:
            insts = self._read_bert_file(insts, path=self.bert_path[path_id])
        return insts

    def _read_bert_file(self, insts, path):
        """
        :param insts:
        :param path:
        :return:
        """
        print("\nRead BERT Features File From {}".format(path))
        now_lines = 0
        with open(path, encoding="utf-8") as f:
            for inst, bert_line in zip(insts, f.readlines()):
                now_lines += 1
                if now_lines % 2000 == 0:
                    sys.stdout.write("\rreading the {} line\t".format(now_lines))
                bert_fea = json.loads(bert_line)
                inst.bert_tokens = bert_fea["features"]["tokens"]
                inst.bert_feature = bert_fea["features"]["values"]
                # print(inst.bert_feature)
            sys.stdout.write("\rReading the {} line\t".format(now_lines))
        return insts



