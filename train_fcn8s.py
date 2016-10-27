#!/usr/bin/env python

import skimage
import skimage.io
import skimage.transform

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

from data import FileIter
from loss import softmaxoutput_loss
from keras.utils import np_utils

import fcn8_vgg
import utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from tensorflow.python.framework import ops

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

#img1 = skimage.io.imread("./test_data/tabby_cat.png")

train_dataiter = FileIter(
    root_dir             = "./",
    flist_name           = "/data/qile/scene_text_preprocessing/mixed.lst",
#        cut_off_size         = 400,
#   flist_name           = "train.lst1",
    rgb_mean             = (123.68, 116.779, 103.939),
    )

with tf.Session() as sess:

    images = tf.placeholder(tf.float32, [None, None, None, 3])
    labels = tf.placeholder(tf.int8, [None, None, None, 2])
    learning_rate = 1e-5
    batch_size = 1

    #feed_dict = {images: img1}
    #batch_images = tf.expand_dims(images, 0)

    vgg_fcn = fcn8_vgg.FCN8VGG()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(images, debug=True, train=True, num_classes=2,random_init_fc8=True)

    print('Finished building Network.')

    logging.warning("Score weights are initialized random.")
    logging.warning("Do not expect meaningful results.")

    logging.info("Start Initializing Variabels.")

    print 'Trainning the Network'
    logits = vgg_fcn.pred_up
    softmax_loss = softmaxoutput_loss(logits, labels, 2) 
    correct_pred = tf.equal(tf.argmax(logits,3), tf.argmax(labels,3))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    saver = tf.train.Saver()

    init = tf.initialize_all_variables()
    sess.run(init)

    print('Loading stored model')
    saver.restore(sess,'/data/qile/tf_model/5-scenetext-60000')
    print('Loading done')

    step = 1
    for epoch in range(50):
        train_dataiter.reset()
        for data in train_dataiter:
            x = data['data']
            x = np.rollaxis(x, 1, 4)
            y = np_utils.to_categorical(data['softmax_label'].flatten().astype(int), 2)
            y_train = y.reshape(1, x.shape[1],x.shape[2],2)

            sess.run(optimizer, feed_dict={images : x, labels: y_train})
            if(step % 10 == 0):
                total_loss, accuracy = sess.run([loss, acc], feed_dict={images : x, labels: y_train})
                print( "Epoch[%d], "%(epoch) + "Iter " + str(step*batch_size) + \
                        ", Minibatch Loss= " + "{:.6f}".format(total_loss) + ", Acc = " + "{:.6f}".format(accuracy) )
            if(step % 1000 == 0):
                saver.save(sess, ('/data/qile/tf_model/%s-scenetext')%(str(epoch)), global_step=step)
            step = step + 1