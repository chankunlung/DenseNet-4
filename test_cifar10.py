#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 23:22:28 2017

@author: yesu
"""
import os
import numpy as np
import pickle
from keras.callbacks import ModelCheckpoint
from rme.callbacks import Step
from densenet import DenseNet

from keras.optimizers import SGD
from keras.utils import np_utils


if __name__ == '__main__':
    X_train = np.load('./data/xtrain.npy')
    y_train = np.load('./data/ytrain.npy')
    X_test = np.load('./data/xtest.npy')
    y_test = np.load('./data/ytest.npy')
    nb_classes = len(np.unique(y_train))
    img_dim = X_train.shape[1:]
    
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    X = np.vstack((X_train, X_test))
    for i in range(img_dim[2]):
        mean = np.mean(X[:, :, :, i])
        std = np.std(X[:, :, :, i])
        X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
        X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std
    
    # set default parameters
    batch_size = 64
    nb_epoch = 300
    depth = 40
    nb_dense_block = 3 
    nb_filter = 16
    growth_rate = 12
    dropout_rate = 0.2
    learning_rate = 0.1
    weight_decay = 1e-4
    
    model = DenseNet(10, img_dim, depth, nb_dense_block, growth_rate, 
                     nb_filter, dropout_rate, weight_decay)
    
    opt = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=["accuracy"])
    
    offset = 0
    save_path = os.path.join('./log','densenet')
    callbacks = []
    callbacks.append(ModelCheckpoint(save_path + '.h5'))
    steps = [nb_epoch/2 - offset, 3*nb_epoch/4 - offset]
    schedule = Step(steps, [learning_rate, 0.1*learning_rate, 0.01*learning_rate], verbose=1)
    callbacks.append(schedule)
    
    hist1 = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=0, validation_data=(X_test, Y_test),
                  callbacks=callbacks, shuffle=True)
    
    with open('./log/cifar10_hist', 'wb') as file:
        pickle.dump(hist1.history, file)
