#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@author: leven03
@contact: xuwang37@163.com
@file: deltaencoder.py
@time: 2020/1/9
@desc:
'''
import torch
import torch.nn as nn
import numpy as np
import random
import os
from linear_classifier import linear_classifier

class Encoder(nn.Module):
    def __init__(self, feature_dim=256, encoder_size=[8192], z_dim=16, dropout=0.5, dropout_input=0.0, leak=0.2):
        super(Encoder, self).__init__()
        self.first_linear = nn.Linear(feature_dim*2, encoder_size[0])

        linear = []
        for i in range(len(encoder_size) - 1):
            linear.append(nn.Linear(encoder_size[i], encoder_size[i+1]))
            linear.append(nn.LeakyReLU(leak))
            linear.append(nn.Dropout(dropout))

        self.linear = nn.Sequential(*linear)
        self.final_linear = nn.Linear(encoder_size[-1], z_dim)
        self.lrelu = nn.LeakyReLU(leak)
        self.dropout_input = nn.Dropout(dropout_input)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, reference_features):
        features = self.dropout_input(features)
        x = torch.cat([features, reference_features], 1)

        # print("features shape is:", features.shape, reference_features.shape)
        # print(x.shape)

        x = self.first_linear(x)
        x = self.linear(x)

        x = self.final_linear(x)

        return x

class Decoder(nn.Module):
    def __init__(self, feature_dim=256, decoder_size=[8192], z_dim=16, dropout=0.5, leak=0.2):
        super(Decoder, self).__init__()
        self.first_linear = nn.Linear(z_dim+feature_dim, decoder_size[0])

        linear = []
        for i in range(len(decoder_size) - 1):
            linear.append(nn.Linear(decoder_size[i], decoder_size[i+1]))
            linear.append(nn.LeakyReLU(leak))
            linear.append(nn.Dropout(dropout))

        self.linear = nn.Sequential(*linear)

        self.final_linear = nn.Linear(decoder_size[-1], feature_dim)
        self.lrelu = nn.LeakyReLU(leak)
        self.dropout = nn.Dropout(dropout)

    def forward(self, reference_features, code):
        x = torch.cat([reference_features, code], 1)

        x = self.first_linear(x)
        x = self.linear(x)

        x = self.final_linear(x)

        return x

# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(EncoderDecoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, features, reference_features):
#         code = self.encoder(features, reference_features)
#         x_hat = self.decoder(reference_features, code)
#
#         return x_hat

class DeltaEncoder(object):
    def __init__(self, args, features, labels, features_test, labels_test, episodes, resume = ''):
        self.count_data = 0
        self.num_epoch = args['num_epoch']
        self.noise_size = args['noise_size']
        self.nb_val_loop = args['nb_val_loop']
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.batch_size = args['batch_size']
        self.drop_out_rate = args['drop_out_rate']
        self.drop_out_rate_input = args['drop_out_rate_input']
        self.best_acc = 0.0
        self.name = args['data_set']
        self.last_file_name = ""
        self.nb_fake_img = args['nb_img']
        self.learning_rate = args['learning_rate']
        self.decay_factor = 0.9
        self.num_shots = args['num_shots']
        self.num_ways = args['num_ways']
        self.resume = resume
        self.save_var_dict = {}

        self.features, self.labels = features, labels
        self.features_test, self.labels_test = features_test, labels_test
        self.episodes = episodes

        self.features_dim = self.features.shape[1]
        self.reference_features = self.random_pairs(self.features, self.labels)

        # discriminator input => image features

        self._create_model()

     # assign pairs with the same labels
    def random_pairs(self,X, labels):
        Y = X.copy()
        for l in range(labels.shape[1]):
            inds = np.where(labels[:,l])[0]
            inds_pairs = np.random.permutation(inds)
            Y[inds,:] = X[inds_pairs,:]
        return Y

    def _create_model(self):
        self.encoder = Encoder(self.features_dim, self.encoder_size, self.noise_size, self.drop_out_rate, self.drop_out_rate_input)
        self.decoder = Decoder(self.features_dim, self.decoder_size, self.noise_size, self.drop_out_rate)

        self.encoder = self.encoder.cuda()
        self.decoder = self.decoder.cuda()

    def loss(self, features_batch, reference_features_batch):
        l1loss = nn.L1Loss().cuda()

        self.pred_noise = self.encoder(features_batch, reference_features_batch)
        self.pred_x = self.decoder(reference_features_batch, self.pred_noise)

        # assert self.pred_noise.shape == self.pred_x.shape
        abs_diff = l1loss(features_batch, self.pred_x)

        w = torch.pow(abs_diff, 2)
        w = w / torch.norm(w)

        loss = w * abs_diff

        return loss

    def optimizer(self, encoder, decoder, lr):
        optimizer = torch.optim.Adam([{'params': encoder.parameters()},
                                      {'params': decoder.parameters()}], lr=lr)

        return optimizer

    def next_batch(self, start, end):
        if start == 0:
            if self.num_shots:
                self.reference_features = self.random_pairs(self.features, self.labels)
            idx = np.r_[:self.features.shape[0]]
            random.shuffle(idx)
            self.features = self.features[idx]
            self.reference_features = self.reference_features[idx]
            self.labels = self.labels[idx]
        if end > self.features.shape[0]:
            end = self.features.shape[0]

        return torch.from_numpy(self.features[start:end]), \
               torch.from_numpy(self.reference_features[start:end]), \
               torch.from_numpy(self.labels[start:end])

    def train(self, verbose=False):
        last_loss_epoch = None
        acc = self.val()
        print('Unseen classes accuracy without training: {}'.format(acc))
        print("-----")
        optimizer = self.optimizer(self.encoder, self.decoder, lr=self.learning_rate)

        self.encoder.train()
        self.decoder.train()
        for epoch in range(self.num_epoch):
            mean_loss_e = 0.0
            for count in range(0, self.features.shape[0], self.batch_size):
                features_batch, reference_features_batch, labels_batch = self.next_batch(count, count + self.batch_size)

                with torch.enable_grad():
                    optimizer.zero_grad()
                    loss_e = self.loss(features_batch.cuda(), reference_features_batch.cuda())
                    loss_e.backward()
                    optimizer.step()

                mean_loss_e += loss_e

                c = (count/self.batch_size) + 1
                if verbose:
                    if np.mod(c, 10) == 1:
                        print('Batch#{0} Loss {1}'.format(c, mean_loss_e / (c + 1e-7)))

            mean_loss_e /= (self.features.shape[0] / self.batch_size)
            if verbose:
                print('epoch : {}: E : {}'.format(epoch, mean_loss_e))
            if last_loss_epoch is not None and mean_loss_e > last_loss_epoch:
                self.learning_rate *= self.decay_factor
                if verbose:
                    print("AE learning rate decay: ", self.learning_rate)
            else:
                last_loss_epoch = mean_loss_e

            acc = self.val()
            if acc > self.best_acc:
                if self.best_acc != 0.0:
                    os.remove(self.last_file_name)
                self.best_acc = acc
                self.last_file_name = "model_weights/" + self.name  + '_' \
                                        + str(self.num_shots) + '_shot_' \
                                        + str(np.around(self.best_acc, decimals=2)) + '_acc.ckpt'
                self.save_model(self.encoder, self.decoder, self.last_file_name)
                print('epoch {}: Higher unseen classes accuracy reached: {} (Saved in {})'.format(epoch+1, acc, self.last_file_name))
            else:
                print('epoch {}: Lower unseen classes accuracy reached: {} (<={})'.format(epoch+1, acc,self.best_acc))
            print("-----")
        return self.best_acc

    def generate_samples(self, reference_features_class, labels_class, nb_ex):
        self.encoder.eval()
        self.decoder.eval()
        iterations = 0
        nb_ex = int(nb_ex)
        features = np.zeros((nb_ex * labels_class.shape[0], self.features.shape[1]))
        labels = np.zeros((nb_ex * labels_class.shape[0], labels_class.shape[1]))
        reference_features = np.zeros((nb_ex * labels_class.shape[0], self.reference_features.shape[1]))
        for c in range(labels_class.shape[0]):
            if True:  # sample "noise" from training set
                inds = np.random.permutation(range(self.features.shape[0]))[:nb_ex]

                noise = self.encoder(torch.Tensor(self.features[inds, ...]).cuda(),
                                     torch.Tensor(self.reference_features[inds, ...]).cuda())

            else:
                noise = torch.from_numpy(np.random.normal(0, 1, (nb_ex, self.noise_size))).cuda()
            reference_features_class_tensor = torch.Tensor(np.tile(reference_features_class[c], (nb_ex, 1))).cuda()
            features[c * nb_ex:(c * nb_ex) + nb_ex] = self.decoder(reference_features_class_tensor, noise).cpu().detach().numpy()

            labels[c * nb_ex:(c * nb_ex) + nb_ex] = np.tile(labels_class[c], (nb_ex, 1))
            reference_features[c * nb_ex:(c * nb_ex) + nb_ex] = np.tile(reference_features_class[c], (nb_ex, 1))
        return features, reference_features, labels


    def val(self, verbose=False):
        acc = []

        for episode_data in self.episodes:
            unique_labels_episode = episode_data[1][:, 0, :]

            features, reference_features, labels = [], [], []
            for shot in range(max(self.num_shots, 1)):
                unique_reference_features_test = episode_data[0][:, shot, :]
                features_, reference_features_, labels_ = self.generate_samples(unique_reference_features_test,
                                                                                unique_labels_episode,
                                                                                self.nb_fake_img / max(self.num_shots, 1))
                features.append(unique_reference_features_test)
                reference_features.append(unique_reference_features_test)
                labels.append(unique_labels_episode)
                features.append(features_)
                reference_features.append(reference_features_)
                labels.append(labels_)
                if verbose:
                    print(np.mean([np.linalg.norm(x) for x in unique_reference_features_test]))
                    print(np.mean([np.linalg.norm(x) for x in features_]))

            features = np.concatenate(features)
            reference_features = np.concatenate(reference_features)
            labels = np.concatenate(labels)

            lin_model = linear_classifier(features, labels, self.features_test,
                                          self.labels_test)
            acc_ = lin_model.learn()
            acc.append(acc_)

        acc = 100 * np.mean(acc)
        return acc


    def save_model(self, encoder, decoder, save_dir):

        if not os.path.exists(os.path.dirname(save_dir)):
            os.mkdir(os.path.dirname(save_dir))
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            }, save_dir)