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

#Model
from model import Networks

EPS = 1e-8


class A3C:

    def __init__(self):

        self.env = gym.make("BreakoutNoFrameskip-v4")
        self.n_act = self.env.action_space.n
        self.agent_history_length = 4
        #self.model = Networks(self.n_act, self.agent_history_length)


    def produce_entropy(self, logits, prob):


        #labels = [tf.one_hot(prob, depth=self.n_act)]
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=prob, logits=logits, axis=1)
        #entropy = tf.reduce_sum(tf.one_hot(pi, depth=self.n_act) * tf.nn.log_softmax(logits), axis=1)

        return entropy


    def sample_action(self, logits):

        prob = tf.nn.softmax(logits)
        log_prob = tf.nn.log_softmax(logits)
        action = tf.random.categorical(prob, 1)[0, 0]



        return action, prob, log_prob











