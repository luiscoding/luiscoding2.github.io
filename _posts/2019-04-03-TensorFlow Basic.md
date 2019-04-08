---
layout:     post
title:      Tensorflow Basic
subtitle:   
date:       2019-04-03
author:     Lu Zhang
header-img: img/images/img-190403.jpg
catalog: true
tags:
    - Machine Learning
    - Tensorflow
---
## What is Tensorflow?
Tensorflow is created by GoogleBrain, with the copyright Apache 2.0, 
Tensorflow is named after the process of the data in DNN, data is represented as tensor, which includes the multiple dimensions 
Tensorflow is used for Goolge Translation, Google Search, etc.

## What we can do with Tensorflow?
We have the version of TF with python, C++, java, and Back Propagating is implemented  with python. So, most of people use python API to train the TF model. 
## Examples Tensorflow 
Example: 
```python
import tensorflow as tf
hw = tf.constant("Hello World")
with tf.Session() as sess:
 print(sess.run(hw))
```
Gradient Decent
```python 
import tensorflow as tf

# Build a dataflow graph.
filename_queue = tf.train.string_input_producer(['1.txt'],num_epochs=1)
reader = tf.TextLineReader()
key,value = reader.read(filename_queue)
num = tf.decode_csv(value,record_defaults=[[0]])
x = tf.Variable([0])
loss = x * num
grads = tf.gradients([loss],x)
grad_x = grads[0]

def train_fn(sess):
  train_fn.counter += 1
  result = sess.run(grad_x)
  print("step %d: grad = %g" % (train_fn.counter,result))

train_fn.counter = 0

sv = tf.train.Supervisor()
tf.train.basic_train_loop(sv,train_fn)
```

