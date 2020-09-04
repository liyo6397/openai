import unittest
from DQN import Agent, DQNetwork
import gym
import numpy as np
import tensorflow as tf
#import universe # register the universe environments




class Test_Qlearning(unittest.TestCase):



    def test_Qnetwork(self):

        episodes = 100
        print_interval = 10
        QL = Agent()

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
        timesteps_per_episode = 100
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
        timesteps_per_episode = 100
        print_interval = 10
        batch_size = 32

        env = gym.make("Taxi-v3")
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        agent = Agent(env, optimizer)

        QL = DQNetwork(agent)
        QL.make_video(env, agent)

        #QL.train(episodes, timesteps_per_episode, print_interval, batch_size)













