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
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop
from gym.spaces import Box, Discrete

EPS = 1e-8



"""
Policies
"""
class actor_critic:
    def __init__(self, env, state):

        self.states = state
        self.n_state, self.n_act = env.observation_space.n, env.action_space.n

    def network(self, inputs, hidden_sizes=(32,), activation='tanh', output_activation=None):

        layer = Dense(self.n_state, activation=activation)(inputs)
        for h in hidden_sizes[1:-1]:
            layer = Dense(h, activation=activation)(layer)
        layer = Dense(hidden_sizes[-1], activation=output_activation)(layer)

        print("Output Size: ", hidden_sizes[-1])

        return layer



    def gaussian_likelihood(self, x, mu, log_std):
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)



    def data_std(self):
        log_std_ini = tf.Variable(-0.5 * np.ones(self.n_act, dtype=np.float32))
        log_std = log_std_ini.read_value()

        std = tf.exp(log_std)

        return std, log_std


    def sample_DiscreteActions(self, inputs, hidden_sizes=(32,), activation='tahn', output_activation=None):

        inputs = np.reshape(inputs, [1,1])
        logits = self.network(inputs, list(hidden_sizes)+[self.n_act], activation, output_activation)
        v = tf.squeeze(self.network(list(hidden_sizes) + [1], activation=activation, output_activation=None), axis=1)
        pi = tf.squeeze(tf.random.categorical(logits, 1), axis=1)

        prob = tf.nn.softmax(logits)
        prob_pi = tf.reduce_sum(tf.one_hot(pi, depth=self.n_act) * prob, axis=1)

        return pi, prob_pi



