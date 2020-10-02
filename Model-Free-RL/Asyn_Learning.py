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
class A3C:
    def __init__(self, env, state):

        self.state = state
        self.n_state, self.n_act = env.observation_space.n, env.action_space.n

    def network(self, hidden_sizes=(32,), activation='tanh', output_activation=None):

        model = Sequential()
        model.add(tf.keras.Input(shape=(self.n_state,)))
        for h in hidden_sizes[:-1]:
            model.add(Dense(h, activation=activation))
        model.add(Dense(hidden_sizes[-1], activation=output_activation))
        #return model.add(Dense(32, activation=output_activation))
        '''model.add(tf.keras.Input(shape=(16,)))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        # Now the model will take as input arrays of shape (None, 16)
        # and output arrays of shape (None, 32).
        # Note that after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(tf.keras.layers.Dense(32))'''
        return model

    def gaussian_likelihood(self, x, mu, log_std):
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    def diagonal_gaussian_kl(self, mu0, log_std0, mu1, log_std1):
        """
        tf symbol for mean KL divergence between two batches of diagonal gaussian distributions,
        where distributions are specified by means and log stds.
        (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)
        """
        var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
        pre_sum = 0.5 * (((mu1 - mu0) ** 2 + var0) / (var1 + EPS) - 1) + log_std1 - log_std0
        all_kls = tf.reduce_sum(pre_sum, axis=1)
        return tf.reduce_mean(all_kls)

    def LogStd(self):
        log_std_ini = tf.Variable(-0.5 * np.ones(self.n_act, dtype=np.float32))
        log_std = log_std_ini.read_value()

        return log_std




    def gaussian_policy(self, a, last_a, hidden_sizes, activation, output_activation):

        mu = self.network(list(hidden_sizes)+[self.n_act], activation, output_activation)
        log_std = self.LogStd()
        std = tf.exp(log_std)
        #Sample actions from policy given states
        pi = mu + tf.random.normal(tf.shape(mu)) * std
        logp = self.gaussian_likelihood(a, mu, log_std)
        logp_pi = self.gaussian_likelihood(pi, mu, log_std)

        #old_mu_ph, old_log_std_ph = placeholders(act_dim, act_dim)
        old_mu, old_log_std = last_a, last_a
        d_kl = self.diagonal_gaussian_kl(mu, log_std, old_mu, old_log_std)

        info = {'mu': mu, 'log_std': log_std}
        info_phs = {'mu': old_mu, 'log_std': old_log_std}

        return pi, logp, logp_pi, info, info_phs, d_kl


    def actor_critic(self, x, a, hidden_sizes=(64,64), activation=tf.tanh,
                         output_activation=None, action_space=None):

        # default policy builder depends on action space
        policy_outs = self.mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation)
        pi, logp, logp_pi, info, info_phs, d_kl = policy_outs

        # value function
        v = tf.squeeze(self.network(list(hidden_sizes)+[1], activation=activation, output_activation = None), axis=1)

        return pi, logp, logp_pi, info, info_phs, d_kl, v
