from typing import List
import numpy as np
import tensorflow as tf
import gym
import random
from collections import deque
import progressbar
from gym import wrappers
import os

#tensorflow
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop


#Gym environment:
#env = gym.make('FrozenLake-v0')
#env = gym.make("Taxi-v3")
#env = gym.make('Cartpole-v0')

class Agent:

    def __init__(self, env, optimizer):
        # setup basic:

        # training
        self.env = env
        self.n_state, self.n_act = env.observation_space.n, env.action_space.n
        self.epsilon = 0.1
        self.expirience_replay = deque(maxlen=2000)
        self.optimizer = optimizer

        # Q table
        self.discount_factor = 0.6  # discount factor
        self.exploration_prob = lambda episode: 50. / (episode + 10)

        #Network
        self.q_network = self.network()
        self.q_target_network = self.network()


    def network(self):

        model = Sequential()
        model.add(Embedding(self.n_state, 10, input_length=1))
        model.add(Reshape((10,)))
        model.add(Dense(10, activation='relu'))
        #model.add(Dense(10, activation='relu'))
        model.add(Dense(self.n_act, activation='linear'))

        model.compile(loss='mse', optimizer=self.optimizer)

        return model



    def act(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            return env.action_space.sample()  # Explore action space

        q_values = self.q_network.predict(state)  # exploit action space
        action = np.argmax(q_values[0])

        return action

    def retrain(self, batch_size):

        minibatch = random.sample(self.expirience_replay, batch_size)

        for state, action, reward, done, next_state in minibatch:

            target = self.q_network.predict(state)

            #if done:
                #target[0][action] = reward
            #else:
            tar = self.q_target_network.predict(next_state)
            target[0][action] = reward + self.discount_factor*np.amax(tar)

            self.q_network.fit(state, target, epochs= 10, verbose= 0)

    def store(self, state, action, reward, next_state, done):
        self.expirience_replay.append((state, action, reward, next_state, done))

    def alighn_target_model(self):
        self.q_target_network.set_weights(self.q_network.get_weights())

class doubleDQNetwork:

    def __init__(self, agent):

        self.agent = agent
        self.env = agent.env




    def train(self, episodes, timesteps_per_episode, print_interval, batch_size):

        episode_rewards = []

        for epoh in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1,1])
            episode_reward = 0

            done = False

            bar = progressbar.ProgressBar(max_value=timesteps_per_episode / 10,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()

            for step in range(timesteps_per_episode):

                action = self.agent.act(state)

                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward

                next_state = np.reshape(next_state, [1,1])
                self.agent.store(state, action, reward, next_state, done)

                state = next_state




                if len(self.agent.expirience_replay) > batch_size:
                    self.agent.retrain(batch_size)
                self.agent.alighn_target_model()

                if step % 10 == 0:
                    bar.update(step / 10 + 1)

                if done or step == timesteps_per_episode-1:
                    episode_rewards.append(episode_reward)
                    print("Episode " + str(epoh) + ": " + str(episode_reward))
                    break

            bar.finish()




    def make_video(self, env, agent):
        env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
        rewards = 0
        steps = 0
        done = False
        state = env.reset()
        state = np.reshape(state, [1, 1])
        while not done:
            action = agent.act(state)
            observation, reward, done, _ = env.step(action)
            steps += 1
            rewards += reward
        env.close()
        print("Testing steps: {} rewards {}: ".format(steps, rewards))


if __name__ ==  '__main__':

    episodes = 100
    timesteps_per_episode = 10
    print_interval = 10
    batch_size = 32

    env = gym.make("Taxi-v3")
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
    agent = Agent(env, optimizer)

    QL = doubleDQNetwork(agent)

    QL.train(episodes, timesteps_per_episode, print_interval, batch_size)
    QL.make_video(env, agent)








