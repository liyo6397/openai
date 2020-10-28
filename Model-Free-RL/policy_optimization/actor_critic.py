import numpy as np
import tensorflow as tf
import gym
import random
from collections import deque
from gym import wrappers
import os
import time
#tensorflow
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Conv2D
from tensorflow.keras.optimizers import Adam, RMSprop
from gym.spaces import Box, Discrete

EPS = 1e-8



"""
Policies
"""
class A3C(Model):
    def policy(self, inputs):

        inputs = np.reshape(inputs, [1,1])
        logits = self.network(inputs, list(self.hidden_sizes)+[self.n_act])
        v = tf.squeeze(self.network(inputs, list(self.hidden_sizes) + [1], output_activation=None), axis=1)
        pi = tf.squeeze(tf.random.categorical(logits, 1), axis=1)

        pi = tf.nn.softmax(logits)
        #prob_pi = tf.reduce_sum(tf.one_hot(pi, depth=self.n_act) * prob, axis=1)

        return v, pi



