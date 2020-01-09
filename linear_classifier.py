#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@author: leven03
@contact: xuwang37@163.com
@file: linear_classifier.py
@time: 2020/1/9
@desc:
'''
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import accuracy_score
import os


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(LinearClassifier, self).__init__()

        self.linear = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class linear_classifier(object):
    def __init__(self, features_train, labels_train, features_test, labels_test,
                 learning_rate=0.0005, number_epoch=25, batch_size=100):
        self.decay_factor = 0.9

        self.features_test = features_test
        self.labels_test = labels_test
        self.features_train = features_train
        self.labels_train = labels_train

        self.class_idx = np.where(np.sum(self.labels_train, axis=0) != 0)[0]
        self.labels_train = self.labels_train[:, self.class_idx]
        self.labels_test = self.labels_test[:, self.class_idx]
        idx = np.any(self.labels_test, axis=1)
        self.labels_test = self.labels_test[idx]
        self.features_test = self.features_test[idx]

        self.learning_rate = learning_rate
        self.number_epoch = number_epoch
        self.batch_size = batch_size


        # print("label test shape is:", labels_test)

        self.classifier = LinearClassifier(self.features_test.shape[1], self.labels_test.shape[1]).cuda()

    def loss(self, features_batch, labels_batch):
        self.classifier.train()
        features_batch = torch.Tensor(features_batch).cuda()
        labels_batch = torch.LongTensor(labels_batch).cuda()

        cel = nn.CrossEntropyLoss()
        logits = self.classifier(features_batch)

        return cel(logits, torch.argmax(labels_batch, 1))

    def training(self, model, learning_rate):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        return optimizer

    def next_batch(self, start, end):
        if start == 0:
            idx = np.r_[:self.features_train.shape[0]]
            random.shuffle(idx)
            self.features_train = self.features_train[idx]
            self.labels_train = self.labels_train[idx]
        if end > self.features_train.shape[0]:
            end = self.features_train.shape[0]
        return self.features_train[start:end], self.labels_train[start:end]

    def val(self):
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(torch.from_numpy(self.features_test).cuda()).cpu()

        acc = accuracy_score(np.argmax(logits, axis=1), np.argmax(self.labels_test, axis=1))
        return acc

    def learn(self):

        #         self.features_test_temp = self.features_test
        self.learning_rate = 0.001

        best_acc = best_acc_seen = best_acc_unseen = 0.0
        last_loss_epoch = None
        optimizer = self.training(self.classifier, self.learning_rate)
        for i in range(self.number_epoch):
            mean_loss_d = 0.0
            for count in range(0, self.features_train.shape[0], self.batch_size):
                features_batch, labels_batch = self.next_batch(count, count + self.batch_size)
                with torch.enable_grad():
                    optimizer.zero_grad()
                    loss_value = self.loss(features_batch, labels_batch)
                    loss_value.backward()
                    optimizer.step()

                mean_loss_d += loss_value

            mean_loss_d /= (self.features_train.shape[0] / self.batch_size)

            if last_loss_epoch is not None and mean_loss_d > last_loss_epoch:
                self.learning_rate *= self.decay_factor
            else:
                last_loss_epoch = mean_loss_d

            acc = self.val()
            if acc > best_acc:
                best_acc = acc
        return best_acc


