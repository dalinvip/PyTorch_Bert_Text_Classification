# @Author : bamtercelboo
# @Datetime : 2018/8/26 8:30
# @File : trainer.py
# @Last Modify Time : 2018/8/26 8:30
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  trainer.py
    FUNCTION : None
"""

import os
import sys
import time
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as utils
from DataUtils.Optim import Optimizer
from DataUtils.utils import *
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Train(object):
    """
        Train
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        Args of data:
            train_iter : train batch data iterator
            dev_iter : dev batch data iterator
            test_iter : test batch data iterator
        Args of train:
            model : nn model
            config : config
        """
        print("Training Start......")
        # for k, v in kwargs.items():
        #     self.__setattr__(k, v)
        self.train_iter = kwargs["train_iter"]
        self.dev_iter = kwargs["dev_iter"]
        self.test_iter = kwargs["test_iter"]
        self.model = kwargs["model"]
        self.config = kwargs["config"]
        self.early_max_patience = self.config.early_max_patience
        self.optimizer = Optimizer(name=self.config.learning_algorithm, model=self.model, lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay, grad_clip=self.config.clip_max_norm)
        self.loss_function = self._loss(learning_algorithm=self.config.learning_algorithm)
        print(self.optimizer)
        print(self.loss_function)
        self.best_score = Best_Result()
        self.train_iter_len = len(self.train_iter)

    @staticmethod
    def _loss(learning_algorithm):
        """
        :param learning_algorithm:
        :return:
        """
        if learning_algorithm == "SGD":
            loss_function = nn.CrossEntropyLoss(reduction="sum")
            return loss_function
        else:
            loss_function = nn.CrossEntropyLoss(reduction="mean")
            return loss_function

    def _clip_model_norm(self, clip_max_norm_use, clip_max_norm):
        """
        :param clip_max_norm_use:  whether to use clip max norm for nn model
        :param clip_max_norm: clip max norm max values [float or None]
        :return:
        """
        if clip_max_norm_use is True:
            gclip = None if clip_max_norm == "None" else float(clip_max_norm)
            assert isinstance(gclip, float)
            utils.clip_grad_norm_(self.model.parameters(), max_norm=gclip)

    def _dynamic_lr(self, config, epoch, new_lr):
        """
        :param config:  config
        :param epoch:  epoch
        :param new_lr:  learning rate
        :return:
        """
        if config.use_lr_decay is True and epoch > config.max_patience and (
                epoch - 1) % config.max_patience == 0 and new_lr > config.min_lrate:
            new_lr = max(new_lr * config.lr_rate_decay, config.min_lrate)
            set_lrate(self.optimizer, new_lr)
        return new_lr

    def _decay_learning_rate(self, config, epoch, init_lr):
        """lr decay 

        Args:
            epoch: int, epoch 
            init_lr:  initial lr
        """
        if config.use_lr_decay:
            lr = init_lr / (1 + self.config.lr_rate_decay * epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return self.optimizer

    def _optimizer_batch_step(self, config, backward_count):
        """
        :return:
        """
        if backward_count % config.backward_batch_size == 0 or backward_count == self.train_iter_len:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _early_stop(self, epoch):
        """
        :param epoch:
        :return:
        """
        best_epoch = self.best_score.best_epoch
        if epoch > best_epoch:
            self.best_score.early_current_patience += 1
            print("Dev Has Not Promote {} / {}".format(self.best_score.early_current_patience, self.early_max_patience))
            if self.best_score.early_current_patience >= self.early_max_patience:
                print("Early Stop Train. Best Score Locate on {} Epoch.".format(self.best_score.best_epoch))
                exit()

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
        return word, mask, sentence_length, labels, batch_size

    def _calculate_loss(self, feats, labels):
        """
        Args:
            feats: size = (batch_size, seq_len, tag_size)
            labels: size = (batch_size, seq_len)
        """
        loss_value = self.loss_function(feats, labels)
        return loss_value

    def train(self):
        """
        :return:
        """
        epochs = self.config.epochs
        clip_max_norm_use = self.config.clip_max_norm_use
        clip_max_norm = self.config.clip_max_norm
        new_lr = self.config.learning_rate

        for epoch in range(1, epochs + 1):
            print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, epochs))
            new_lr = self._dynamic_lr(config=self.config, epoch=epoch, new_lr=new_lr)
            # self.optimizer = self._decay_learning_rate(config=self.config, epoch=epoch - 1, init_lr=self.config.learning_rate)
            print("now lr is {}".format(self.optimizer.param_groups[0].get("lr")), end="")
            start_time = time.time()
            random.shuffle(self.train_iter)
            self.model.train()
            steps = 1
            backward_count = 0
            self.optimizer.zero_grad()
            for batch_count, batch_features in enumerate(self.train_iter):
                backward_count += 1
                # self.optimizer.zero_grad()
                word, mask, sentence_length, labels, batch_size = self._get_model_args(batch_features)
                logit = self.model(word, sentence_length, train=True)
                loss = self._calculate_loss(logit, labels)
                loss.backward()
                self._clip_model_norm(clip_max_norm_use, clip_max_norm)
                self._optimizer_batch_step(config=self.config, backward_count=backward_count)
                # self.optimizer.step()
                steps += 1
                if (steps - 1) % self.config.log_interval == 0:
                    accuracy = self.getAcc(logit, labels, batch_size)
                    sys.stdout.write(
                        "\nbatch_count = [{}] , loss is {:.6f}, [accuracy is {:.6f}%]".format(batch_count + 1, loss.item(), accuracy))
            end_time = time.time()
            print("\nTrain Time {:.3f}".format(end_time - start_time), end="")
            self.eval(model=self.model, epoch=epoch, config=self.config)
            self._model2file(model=self.model, config=self.config, epoch=epoch)
            self._early_stop(epoch=epoch)

    def eval(self, model, epoch, config):
        """
        :param model: nn model
        :param epoch:  epoch
        :param config:  config
        :return:
        """
        eval_start_time = time.time()
        self.eval_batch(self.dev_iter, model, self.best_score, epoch, config, test=False)
        eval_end_time = time.time()
        print("Dev Time {:.3f}".format(eval_end_time - eval_start_time))

        eval_start_time = time.time()
        self.eval_batch(self.test_iter, model, self.best_score, epoch, config, test=True)
        eval_end_time = time.time()
        print("Test Time {:.3f}".format(eval_end_time - eval_start_time))

    def _model2file(self, model, config, epoch):
        """
        :param model:  nn model
        :param config:  config
        :param epoch:  epoch
        :return:
        """
        if config.save_model and config.save_all_model:
            save_model_all(model, config.save_dir, config.model_name, epoch)
        elif config.save_model and config.save_best_model:
            save_best_model(model, config.save_best_model_path, config.model_name, self.best_score)
        else:
            print()

    def eval_batch(self, data_iter, model, best_score, epoch, config, test=False):
        """
        :param data_iter:  eval batch data iterator
        :param model: eval model
        :param best_score:
        :param epoch:
        :param config: config
        :param test:  whether to test
        :return: None
        """
        model.eval()
        # eval time
        corrects = 0
        size = 0
        for batch_features in data_iter:
            word, mask, sentence_length, labels, batch_size = self._get_model_args(batch_features)
            logit = self.model(word, sentence_length, train=False)
            size += batch_features.batch_length
            corrects += (torch.max(logit, 1)[1].view(labels.size()).data == labels.data).sum()

        assert size is not 0, print("Error")
        accuracy = float(corrects) / size * 100.0

        test_flag = "Test"
        if test is False:
            print()
            test_flag = "Dev"
            best_score.current_dev_score = accuracy
            if accuracy >= best_score.best_dev_score:
                best_score.best_dev_score = accuracy
                best_score.best_epoch = epoch
                best_score.best_test = True
        if test is True and best_score.best_test is True:
            best_score.p = accuracy
        print("{} eval: Accuracy = {:.6f}%".format(test_flag, accuracy))
        if test is True:
            print("The Current Best Dev Accuracy: {:.6f}, Locate on {} Epoch.".format(best_score.best_dev_score,
                                                                                      best_score.best_epoch))
            print("The Current Best Test Accuracy: accuracy = {:.6f}%".format(best_score.p))
        if test is True:
            best_score.best_test = False

    @staticmethod
    def getAcc(logit, target, batch_size):
        """
        :param logit:  model predict
        :param target:  gold label
        :param batch_size:  batch size
        :param config:  config
        :return:
        """
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = float(corrects) / batch_size * 100.0
        return accuracy







