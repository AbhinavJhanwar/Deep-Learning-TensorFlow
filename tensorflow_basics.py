# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 09:46:40 2018

@author: abhinav.jhanwar
"""

import tensorflow as tf

###############################################################################
# Constants takes no input, you use them to store constant values. 
# They produce a constant output that it stores.
a = tf.constant(2.0)
b = tf.constant(3.0)
c = a * b

sess = tf.Session()
sess.run(c)
###############################################################################

###############################################################################
# Placeholders allow you to feed input on the run. 
# Because of this flexibility, placeholders are used which allows your 
# computational graph to take inputs as parameters. 
# Defining a node as a placeholder assures that node, that it is expected to 
# receive a value later or during runtime. 
# Here, "runtime" means that the input is fed to the placeholder when you run 
# your computational graph.

# Creating placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Assigning addition operation w.r.t. a and b to node add
add = a + b

# Create session object
sess = tf.Session()

# Executing add by passing the values [1, 3] [2, 4] for a and b respectively
output = sess.run(add, {a: [1,3], b: [2, 4]})
print('Adding a and b:', output)
###############################################################################

###############################################################################
# Variables allow you to modify the graph such that it can produce new outputs
# with respect to the same inputs.
# A variable allows you to add such parameters or node to the graph that are
# trainable. That is, the value can be modified over the period of a time.

# Variables are defined by providing their initial value and type
variable = tf.Variable([0.9,0.7], dtype = tf.float32)

# variable must be initialized before a graph is used for the first time. 
init = tf.global_variables_initializer()
sess.run(init)