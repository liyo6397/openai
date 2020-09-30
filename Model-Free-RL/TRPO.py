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

EPS = 1e-8

"""
Policies
"""
class Policies:
    def __init__(self, state, env):

        self.state = state
        self.n_state, self.n_act = env.observation_space.n, env.action_space.n

    def network(self, hidden_sizes=(32,), activation='tanh', output_activation=None):

        #model = Sequential()
        for h in hidden_sizes[:-1]:
            Dense(self.n_state, units=h, activation=activation)
        return Dense(self.n_state, units=hidden_sizes[-1], activation=output_activation)

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


    def mlp_gaussian_policy(self, a, last_a, hidden_sizes, activation, output_activation):

        mu = self.network(list(hidden_sizes)+[self.n_act], activation, output_activation)
        log_std = tf.Variable(name='log_std', initializer=-0.5*np.ones(self.n_act, dtype=np.float32))
        std = tf.exp(log_std)
        pi = mu + tf.random.normal(tf.shape(mu)) * std
        logp = self.gaussian_likelihood(a, mu, log_std)
        logp_pi = self.gaussian_likelihood(pi, mu, log_std)

        #old_mu_ph, old_log_std_ph = placeholders(act_dim, act_dim)
        old_mu, old_log_std = last_a, last_a
        d_kl = self.diagonal_gaussian_kl(mu, log_std, old_mu, old_log_std)

        info = {'mu': mu, 'log_std': log_std}
        info_phs = {'mu': old_mu, 'log_std': old_log_std}

        return pi, logp, logp_pi, info, info_phs, d_kl



class TRPO:
    def __init__(self, env,  seed=0,
             steps_per_epoch=4000, epochs=50, gamma=0.99, delta=0.01, residual_tol=1e-3,
             train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
             backtrack_coeff=0.8, lam=0.97, max_ep_len=1000,
             save_freq=10):

        self.env = env
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.delta = delta
        self.residual_tol = residual_tol
        self.train_v_iters = train_v_iters
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq

        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.shape
        self.action_space = env.action_space

    def train(self):
        start_time = time.time()
        state, ep_ret, ep_len = self.env.reset(), 0, 0
        state = np.reshape(state, [1, 1])

        last_action = None
        for epoch in range(self.epochs):

            action = self.env.action_space

            next_state, reward, done, info = self.env.step(action)
            ep_ret += reward
            ep_len += 1








