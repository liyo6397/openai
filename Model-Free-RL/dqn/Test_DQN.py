import unittest
#from DQN import Agent, DQNetwork
from doubleDQN import doubleDQN
import gym
import numpy as np
import tensorflow as tf

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

        q_network = self.dqn.network2()
        action = q_network.predict(states)

        print("Action: ", action)



    def test_expReplay(self):
        env = gym.make("Taxi-v3")
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        QL = doubleDQN(env, optimizer)

        print(QL.expirience_replay)

    def test_timestep_episode(self):

        episodes = 100
        timesteps_per_episode = 50
        print_interval = 10
        batch_size = 32

        env = gym.make("Taxi-v3")
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1)



        QL = doubleDQN(env, optimizer)

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
        #agent = Agent(env, optimizer)

        QL = doubleDQN(env, optimizer)
        QL.train(episodes, timesteps_per_episode, print_interval, batch_size)

