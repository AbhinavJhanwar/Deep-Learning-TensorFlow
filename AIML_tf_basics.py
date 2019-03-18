# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:37:04 2019

@author: abhinav.jhanwar
"""

''' Adventures In Machine Learning '''

import tensorflow as tf
import numpy as np

################################################################
# implementing equation - a=(b+c)∗(c+2) using tf

# first, create a TensorFlow constant and variables
# initialize a value, optionally assign a name to constant/variable, optionally assign type of constant/variable
const = tf.constant(2.0, name="const")
b = tf.Variable(2.0, name='b')
c = tf.Variable(1.0, name='c')
# these constant and variable are only created when initialize command is run

# now create some operations i.e. operations
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

# setup the variable initialisation
init_op = tf.global_variables_initializer()

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    # compute the output of the graph
    a_out = sess.run(a)
    print("Variable a is {}".format(a_out))
###################################################################

#####################################################################
# using an array of variable b

const = tf.constant(2.0, name="const")
c = tf.Variable(1.0, name='c')

# type of data in array, shape, name
b = tf.placeholder(tf.float32, [None, 1], name='b')    

# now create some operations i.e. operations
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

# setup the variable initialisation
init_op = tf.global_variables_initializer()

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    # compute the output of the graph
    a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    print("Variable a is {}".format(a_out))
    train_writer = tf.summary.FileWriter('Output', sess.graph)
    #train_writer.add_graph(sess.graph)
    # to view graph run following command - tensorboard --logdir='location of graph'
    
    
    
    
    
    
    
    