import unittest
#from DQN import Agent, DQNetwork
#from DoubleDQN import doubleDQNetwork
import gym
import numpy as np
import tensorflow as tf
from Asyn_Learning import actor_critic
#import universe # register the universe environments
from tensorflow.keras.layers import Dense, Embedding, Reshape

class Test_AsynLearning(unittest.TestCase):

    def test_network(self):
        env = gym.make("Taxi-v3")
        state = env.reset()

        a3c = actor_critic(env, state)
        hidden_sizes = (4, 4)
        activation = tf.tanh
        output_activation = None

        inputs = tf.ones((3,3))
        mu = a3c.network(inputs, hidden_sizes, activation, output_activation)
        print("mu: ",mu)

    def test_logits(self):

        env = gym.make("Taxi-v3")
        states = env.reset()

        a3c = actor_critic(env, states)
        hidden_sizes = (32, 32)
        activation = tf.tanh
        output_activation = None

        inputs = np.reshape(states, [1,1])
        logits = a3c.network(inputs, list(hidden_sizes)+[env.action_space.n], activation, output_activation)

        print(logits)



    def test_data_logstd(self):
        env = gym.make("Taxi-v3")
        state = env.reset()

        a3c = A3C(env, state)

        std, log_std = a3c.data_std()

        print("shape: ", std.shape)
        print("std: ",std)
        print("Log std: ", log_std)

    def test_pi(self):

        env = gym.make("Taxi-v3")
        states = env.reset()

        a3c = actor_critic(env, states)
        hidden_sizes = (32, 32)
        activation = tf.tanh
        output_activation = None

        inputs = np.reshape(states, [1, 1])
        logits = a3c.network(inputs, list(hidden_sizes) + [env.action_space.n], activation, output_activation)

        pi = tf.squeeze(tf.random.categorical(logits, 1), axis=1)

        print("pi: ", pi)

    def test_discrete_prob(self):

        env = gym.make("Taxi-v3")
        states = env.reset()

        action = env.action_space.sample()
        n_act = env.action_space.n

        a3c = actor_critic(env, states)
        hidden_sizes = (32, 32)
        activation = tf.tanh
        output_activation = None

        inputs = np.reshape(states, [1,1])
        logits = a3c.network(inputs, list(hidden_sizes) + [env.action_space.n], activation, output_activation)
        pi = tf.squeeze(tf.random.categorical(logits, 1), axis=1)

        prob = tf.nn.softmax(logits)
        prob_pi = tf.reduce_sum(tf.one_hot(pi, depth=n_act) * prob, axis=1)


        print("prob: ", prob)
        print("prob_pi: ", prob_pi)



    def test_mlp_gaussian_policy(self):

        env = gym.make("Taxi-v3")
        state = env.reset()

        a3c = A3C(env,state)
        action = env.action_space
        last_action = None
        hidden_sizes = (32, 32)
        activation = tf.tanh
        output_activation = None

        pi, logp, logp_pi, info, info_phs, d_kl = a3c.gaussian_policy(action, last_action, hidden_sizes, activation, output_activation)

        print("pi: ", pi)
        print("logp: ", logp_pi)
        print("logp_pi: ", logp_pi)
        print("d_kl: ", d_kl)















