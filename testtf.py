#!/usr/bin/env python
#-*- coding: utf-8 -*-
import tensorflow as tf

"""
File Name: testtf.py
Created Time: 2016-12-12
"""

def hello_tf(s):
    hello = tf.constant("Hello,"+s)
    sess = tf.Session()
    print sess.run(hello) 

if __name__ == '__main__':
  #Todo Somethings
  hello_tf("tf")
