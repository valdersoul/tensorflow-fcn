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
import glob

from data import FileIter
from loss import loss
from keras.utils import np_utils

import fcn8_vgg
import utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

from tensorflow.python.framework import ops

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

#img1 = skimage.io.imread("./test_data/tabby_cat.png")

with tf.Session() as sess:

    images = tf.placeholder(tf.float32, [None, None, None, 3])
    labels = tf.placeholder(tf.int8, [None, 720, 720, 2])
    learning_rate = 1e-6
    batch_size = 1

    #feed_dict = {images: img1}
    #batch_images = tf.expand_dims(images, 0)

    vgg_fcn = fcn8_vgg.FCN8VGG()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(images, debug=False, num_classes=2,random_init_fc8=True)

    print('Finished building Network.')

    logging.warning("Score weights are initialized random.")
    logging.warning("Do not expect meaningful results.")

    logging.info("Start Initializing Variabels.")

    print 'Loading the Network'
    logits = vgg_fcn.pred_up
    softmax_loss = loss(logits, labels, 2) 
    correct_pred = tf.equal(tf.argmax(logits,3), tf.argmax(labels,3))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(softmax_loss, global_step=global_step)
    saver = tf.train.Saver()

    init = tf.initialize_all_variables()
    sess.run(init)

    saver.restore(sess,'/data/qile/tf_model/0my-model-1000')

    imgs = glob.glob('./test_data/*.jpg')
    length = len(imgs)
    count = 0
    for img in imgs:
        count += 1
        print 'processing ...................................%d/%d' %(count, length)
        img1 = skimage.io.imread(img)
        if(img1.shape[0] > 2000 or img1.shape[1] > 2000):
            img1 = skimage.transform.rescale(img1, 0.5)
        h = img1.shape[1]
        w = img1.shape[0]
        img1 = np.expand_dims(img1, axis=0)
        #img1 = np.rollaxis(img1, 1, 4)

        feed_dict = {images: img1}
        tensors = vgg_fcn.pred_up
        up = sess.run(tensors, feed_dict=feed_dict)
        #down = tf.argmax(down, dimension=3)
        up = tf.reshape(up[0], (-1, 2))
        up = tf.nn.softmax(up)
        up = up.eval()[:,1]
        up = up.reshape((w,h))
        up = up > 0.2
        scp.misc.imsave('res/' + img.replace('jpg', 'png')[11:], up)
