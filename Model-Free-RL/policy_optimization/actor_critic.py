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
from utils import Networks

EPS = 1e-8


class A3C:

    def __init__(self):

        self.env = gym.make("BreakoutNoFrameskip-v4")
        self.n_act = self.env.action_space.n
        self.agent_history_length = 4
        self.net = Networks(self.env, self.agent_history_length)


    def get_policy(self, layer_data):

        logits = Dense(units=self.n_act, activation=None)(layer_data)
        prob_pi = tf.keras.activations.softmax(logits)

        return prob_pi



    def critic_value(self, layer_data):

        logits = Dense(units=1, activation=None)(layer_data)

        return logits




