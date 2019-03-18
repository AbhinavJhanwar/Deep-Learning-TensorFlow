# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:37:22 2019

@author: abhinav.jhanwar
"""


''' adding scalar summaries like accuracy '''

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100

with tf.name_scope("inputs"):
    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(tf.float32, [None, 784])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10])

# now you can see the graph is very complex and hence lets go ahead and concise the graph by naming each layer
with tf.name_scope("layer_1"):
    W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([300]), name='b1')
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_logit = tf.nn.relu(hidden_out)


with tf.name_scope("layer_2"):
    # now declare the weights connecting the input to the hidden layer
    W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([10]), name='b2')
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_logit, W2), b2))
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)

with tf.name_scope("cross_entropy"):   
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))

with tf.name_scope("optimiser"):  
    # add an optimiser
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

with tf.name_scope("accuracy"):  
    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # calculate mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# add a summary to store the accuracy
# The first argument is the name you chose to give the quantity within the TensorBoard visualization, 
# and the second is the operation (which must return a single real value) you want to log 
tf.summary.scalar('acc_summary', accuracy)

# create summary of images to understand which images were classified correctly and incorrectly
with tf.variable_scope("getimages"):
    # takes the scaled input tensor (representing the hand written digit images) and 
    # returns only those images which were correctly classified by the network
    correct_inputs = tf.boolean_mask(x, correct_prediction)
    image_summary_true = tf.summary.image('correct_images', tf.reshape(correct_inputs, (-1, 28, 28, 1)),
                                          max_outputs=5)
    
    incorrect_inputs = tf.boolean_mask(x, tf.logical_not(correct_prediction))
    image_summary_false = tf.summary.image('incorrect_images', tf.reshape(incorrect_inputs, (-1, 28, 28, 1)),
                                        max_outputs=5)
    
# create a merger to merge all the summaries
merged = tf.summary.merge_all()

# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   writer = tf.summary.FileWriter('Output', sess.graph)
   
   total_batch = int(len(mnist.train.labels) / batch_size)
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], 
                         feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        acc, summary = sess.run([accuracy, merged], feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Accuracy:", acc)
        writer.add_summary(summary, epoch)
   print("Training Complete!")











