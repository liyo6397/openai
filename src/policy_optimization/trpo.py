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








