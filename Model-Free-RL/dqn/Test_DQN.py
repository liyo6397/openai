import unittest
#from DQN import Agent, DQNetwork
from doubleDQN import doubleDQN
from NNetwork import Networks

import gym
import numpy as np
import tensorflow as tf
import progressbar

class Test_Qlearning(unittest.TestCase):

    def setUp(self):

        self.env = gym.make("Taxi-v3")
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        self.dqn = doubleDQN(self.env, self.optimizer)


    def test_Qnetwork(self):

        episodes = 100
        print_interval = 10
        env = gym.make("Taxi-v3")
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        QL = doubleDQN(env, optimizer)

        model = QL.network2()
        print("Model built: ")
        print(model.summary())

    def test_predict(self):

        states = self.env.reset()
        print("states: ", states)
        states = np.reshape(states, [1, 1])

        q_network = self.dqn.q_network
        action = q_network.predict(states)

        print("Action: ", action)

    def test_get_action(self):

        state = self.env.reset()
        state = np.reshape(state, [1, 1])

        action = self.dqn.get_action(state)

        print(action)

    def test_timestep_episode(self):

        iterations = 100
        max_actions = 50
        print_interval = 10
        batch_size = 32


        self.dqn.train(iterations, max_actions, print_interval, batch_size)

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
        max_action = 500
        print_interval = 10
        batch_size = 10

        env = gym.make("Taxi-v3")
        #env = gym.make('FrozenLake-v0')
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        #agent = Agent(env, optimizer)

        QL = doubleDQN(env, optimizer)
        QL.train(episodes, max_action, print_interval, batch_size)

    def test_NNclass(self):

        state = self.env.reset()
        state = np.reshape(state, [1, 1])

        model = self.dqn.q_network
        q_val = model(state)
        print("Model: ", model)
        print("Q values:", q_val)

    def test_DeepNN_model(self):

        NN = Networks(self.env.action_space.n)

        state = self.env.reset()
        state = np.reshape(state, [1, 1])

        model = NN.DeepNN_model()
        raw = model(state)
        pred = model.predict(state)

        print("Model: ", raw)
        print("Pred: ", pred)


    def test_gradientTape_training(self):

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        dqn = doubleDQN(self.env, optimizer)

        iterations = 100
        max_actions = 50
        print_interval = 10
        batch_size = 32

        dqn.train(iterations, max_actions, print_interval, batch_size)







