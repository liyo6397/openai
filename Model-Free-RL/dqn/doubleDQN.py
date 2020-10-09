import numpy as np
import tensorflow as tf
import gym
import random
from collections import deque
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop


class doubleDQN:

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
        self.learning_rate = 0.00
        self.q_network = self.network()

        #DQN type

        self.q_target_network = self.network()



    def network(self):

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.n_state, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.n_act, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def get_action(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            return self.env.action_space.sample()  # Explore action space

        q_values = self.q_network.predict(state)  # exploit action space
        action = np.argmax(q_values[0])

        return action

    def retrain(self, batch_size):

        minibatch = random.sample(self.expirience_replay, batch_size)

        for state, action, reward, done, next_state in minibatch:

            target = self.q_network.predict(state)

            if done:
                target[0][action] = reward
            else:
                tar = self.q_target_network.predict(next_state)
                target[0][action] = reward + self.discount_factor*np.amax(tar)



            self.q_network.fit(state, target, epochs= 1, verbose= 0)

    def store(self, state, action, reward, next_state, done):
        self.expirience_replay.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.q_target_network.set_weights(self.q_network.get_weights())

    def train(self, episodes, timesteps_per_episode, print_interval, batch_size):
        for epoh in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, 1])

            done = False

            for step in range(timesteps_per_episode):

                action = self.get_action(state)

                next_state, reward, done, info = self.env.step(action)

                next_state = np.reshape(next_state, [1, 1])
                self.store(state, action, reward, next_state, done)

                state = next_state

                if done:
                    self.update_target_model()
                    print("episode number: ", epoh,", reward: ",r , "time score: ", t)
                    break

                if len(self.expirience_replay) > batch_size:
                    self.retrain(batch_size)



            if (epoh + 1) % print_interval == 0:
                print("Episodes: {}".format(epoh + 1))
                self.env.render()
