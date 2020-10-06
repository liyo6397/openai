import unittest
from DQN import Agent, DQNetwork
from DoubleDQN import doubleDQNetwork
import gym
import numpy as np
import tensorflow as tf
from Asyn_Learning import A3C
#import universe # register the universe environments
from tensorflow.keras.layers import Dense, Embedding, Reshape




class Test_Qlearning(unittest.TestCase):


    def test_Qnetwork(self):

        episodes = 100
        print_interval = 10
        env = gym.make("Taxi-v3")
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        QL = Agent(env, optimizer)

        model = QL.network()
        #env = gym.make("Taxi-v3")
        #state = env.reset()

        state = np.arange(32)

        q_values = model.predict(state)

        print("Q values: ", q_values)

    def test_action(self):
        episodes = 100
        print_interval = 10
        QL = Agent()

        state = np.arange(15)

        action = QL.q_network.predict(state)

        print("Action: ", action)
        print(action.shape)

    def test_expReplay(self):

        agent = Agent()

        print(agent.expirience_replay)

    def test_timestep_episode(self):

        episodes = 100
        timesteps_per_episode = 50
        print_interval = 10
        batch_size = 32

        env = gym.make("Taxi-v3")
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        agent = Agent(env,optimizer)


        QL = DQNetwork(agent)

        QL.train(episodes, timesteps_per_episode, print_interval, batch_size)

    def test_gym(self):

        env = gym.make("Taxi-v3")
        observation = env.reset()
        for _ in range(1000):
            env.render()
            action = env.action_space.sample()  # your agent here (this takes random actions)
            observation, reward, done, info = env.step(action)

            if done:
                observation = env.reset()
        env.close()

    def test_video(self):

        episodes = 100
        timesteps_per_episode = 50
        print_interval = 10
        batch_size = 10

        env = gym.make("Taxi-v3")
        #env = gym.make('FrozenLake-v0')
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        agent = Agent(env, optimizer)

        QL = doubleDQNetwork(agent)
        QL.train(episodes, timesteps_per_episode, print_interval, batch_size)
        QL.make_video(env, agent)

class Test_AsynLearning(unittest.TestCase):

    def test_network(self):
        env = gym.make("Taxi-v3")
        state = env.reset()

        a3c = A3C(env, state)
        hidden_sizes = (4, 4)
        activation = tf.tanh
        output_activation = None

        inputs = tf.ones((3,3))
        mu = a3c.network(inputs, hidden_sizes, activation, output_activation)
        print("mu: ",mu)

    def test_mu(self):

        env = gym.make("Taxi-v3")
        states = env.reset()

        a3c = A3C(env, states)
        hidden_sizes = (32, 32)
        activation = tf.tanh
        output_activation = None

        inputs = np.reshape(states, [1,1])
        mu = a3c.network(inputs, list(hidden_sizes)+[env.action_space.n], activation, output_activation)

        print(mu)



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

        a3c = A3C(env, states)
        hidden_sizes = (32, 32)
        activation = tf.tanh
        output_activation = None

        inputs = np.reshape(states, [1, 1])
        mu = a3c.network(inputs, list(hidden_sizes) + [env.action_space.n], activation, output_activation)
        std, log_std = a3c.data_std()

        pi = mu + tf.random.normal(tf.shape(mu)) * std

        print("pi: ", pi)

    def test_gaussian_liklihood(self):

        env = gym.make("Taxi-v3")
        states = env.reset()

        action = env.action_space.sample()

        a3c = A3C(env, states)
        hidden_sizes = (32, 32)
        activation = tf.tanh
        output_activation = None

        inputs = np.reshape(states, [1,1])
        mu = a3c.network(inputs, list(hidden_sizes) + [env.action_space.n], activation, output_activation)
        std, log_std = a3c.data_std()
        pi = mu + tf.random.normal(tf.shape(mu)) * std

        logp = a3c.gaussian_likelihood(action, mu, log_std)
        log_pi = a3c.gaussian_likelihood(pi, mu, log_std)

        print("action: ", action)
        print("log_p: ", logp)
        print("log_pi: ", log_pi)


    def test_diagonal_gaussian_kl(self):
        env = gym.make("Taxi-v3")
        states = env.reset()

        action = env.action_space.sample()

        a3c = A3C(env, states)
        hidden_sizes = (32, 32)
        activation = tf.tanh
        output_activation = None

        inputs = np.reshape(states, [1, 1])
        mu = a3c.network(inputs, list(hidden_sizes) + [env.action_space.n], activation, output_activation)
        std, log_std = a3c.data_std()
        pi = mu + tf.random.normal(tf.shape(mu)) * std

        d_kl = self.diagonal_gaussian_kl(mu, log_std, old_mu, old_log_std)

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















