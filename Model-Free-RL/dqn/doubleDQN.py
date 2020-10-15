import numpy as np
import tensorflow as tf
import gym
import random
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop

import progressbar


class doubleDQN:

    def __init__(self, env, optimizer):
        # setup basic:

        # training
        self.env = env
        self.n_state, self.n_act = env.observation_space.n, env.action_space.n
        self.epsilon = 0.1
        self.expirience_replay = deque(maxlen=2000)
        self.optimizer = optimizer
        self.record = []

        # Q table
        self.discount_factor = 0.6  # discount factor
        self.exploration_prob = lambda episode: 50. / (episode + 10)

        #Network
        self.learning_rate = 0.00
        self.q_network = self.network()
        self.q_target_network = self.network()



    def network(self, units=24):

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(units=units, activation='relu', input_shape=(1,)))
        model.add(Dense(units=units, activation='relu'))
        model.add(Dense(self.n_act, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def network2(self,units=24):

        #try:
        #inputs = tf.keras.Input(shape=(self.n_state))
        #except:
        inputs = tf.keras.Input(shape=(1,))

        outputs = Dense(units=units, activation='relu')(inputs)
        outputs = Dense(units=units, activation='relu')(outputs)
        outputs = Dense(self.n_act, activation='linear')(outputs)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(lr=self.learning_rate),loss='mse')

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



            self.q_network.fit(state, target, verbose= 0)

    def store(self, state, action, reward, next_state, done):
        self.expirience_replay.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.q_target_network.set_weights(self.q_network.get_weights())

    def train(self, episodes, max_actions, print_interval, batch_size):
        for epoh in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, 1])

            epoh_reward = []
            time_score = 0

            done = False

            bar = progressbar.ProgressBar(max_value=max_actions / 10,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()

            for step in range(max_actions):

                action = self.get_action(state)

                next_state, reward, done, info = self.env.step(action)

                next_state = np.reshape(next_state, [1, 1])
                self.store(state, action, reward, next_state, done)

                state = next_state
                epoh_reward.append(reward)
                time_score += 1

                if done:
                    self.update_target_model()
                    ave_reward = np.mean(epoh_reward)
                    print("episode number: ", epoh,", reward: ",ave_reward , "time score: ", time_score)
                    self.record_info(epoh, reward, time_score)
                    break

                if len(self.expirience_replay) > batch_size:
                    self.retrain(batch_size)

                if (step % 50 )==0 and step>0:
                    self.update_target_model()




            bar.finish()
            if (epoh + 1) % print_interval == 0:
                print("Episodes: {}".format(epoh + 1))
                self.env.render()

    def record_info(self, epoh, reward, time_score):

        self.record.append((epoh, reward, time_score))







