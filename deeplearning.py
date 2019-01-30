# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:35:19 2019

@author: leedongjae
"""

import numpy as np
import tensorflow as tf
from model import *



def deeplearning(tr_data, tr_ans, ts_data, ts_ans):
    tr_row = tr_data.shape[0]
    tr_column = tr_data.shape[1]
    ts_row = ts_data.shape[0]
    ts_column = ts_data.shape[1]
    
    X = tf.placeholder(tf.float32, [None, tr_column])
    Y = tf.placeholder(tf.float32, [None, 1])
    
    nh1 = 256
    nh2 = 256
    nh3 = 256
    
    W = {
        'hl1' : tf.Variable(tf.random_normal([tr_column, nh1])),
        'hl2' : tf.Variable(tf.random_normal([nh1, nh2])),
        'hl3' : tf.Variable(tf.random_normal([nh2, nh3])),
        'out' : tf.Variable(tf.random_normal([nh3, 1]))        
            
    }
    
    
    B = {
        'hl1' : tf.Variable(tf.random_normal([nh1])),
        'hl2' : tf.Variable(tf.random_normal([nh2])),
        'hl3' : tf.Variable(tf.random_normal([nh3])),
        'out' : tf.Variable(tf.random_normal([1]))       
    }
    
    HL1 = tf.nn.relu(tf.add(tf.matmul(X, W['hl1'])), B['hl1'])
    HL2 = tf.nn.relu(tf.add(tf.matmul(HL1, W['hl2'])), B['hl2'])
    HL3 = tf.nn.relu(tf.add(tf.matmul(HL2, W['hl3'])), B['hl3'])
    model_LC = tf.nn.softmax(tf.matmul(HL3, W['out']) + B['out'])

    cross_entropy =  - tf.reduce_mean(tf.reduce_sum(Y*tf.log(model_LC), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(cross_entropy)
    
    predict = tf.equal(tf.arg_max(model_LC,1), tf.arg_max(Y,1))
    accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(3000):
            sess.run(train, feed_dict = {
                    X: tr_data, 
                    Y: tr_ans
            })
        
            if epoch % 300 == 0:
                loss_train = cross_entropy.eval(session = sess, feed_dict = {
                        X : tr_data,
                        Y : tr_ans
                })
                accuracy_train = accuracy.eval(session = sess, feed_dict = {
                        X : tr_data,
                        Y : tr_ans
                })
        
            print('epoch:', epoch,
                  'loss of training set = ', loss_train,
                  'accuracu of training set = ', accuracy_train
                  )
            
        correct_prediction = tf.equal(tf.argmax(model_LC, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy_.eval({X: ts_data, Y: ts_ans}))

        
    