import numpy as np
import tensorflow as tf
import gym
import random
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop
from NNetwork import DeepNN_model


import progressbar
from record import summary


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
        self.q_network = DeepNN_model(self.n_act)
        self.q_target_network = DeepNN_model(self.n_act)

        #Summary
        self.record = summary()



    def old_network(self, units=24):

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(units=units, activation='relu', input_shape=(1,)))
        model.add(Dense(units=units, activation='relu'))
        model.add(Dense(self.n_act, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def network(self,units=24):

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




    def get_eps(self, total_step):

        if total_step > 10**4:
            epsilon = 0.1
        else:
            epsilon = 0.01

        return epsilon




    def get_action(self, state, eps):

        if random.uniform(0, 1) <= eps:
            return self.env.action_space.sample()  # Explore action space

        q_values = self.q_network.predict(state)  # exploit action space
        action = np.argmax(q_values[0])

        return action

    def retrain(self, batch_size):

        minibatch = random.sample(self.expirience_replay, batch_size)

        with tf.GradientTape as tape:
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

    def train(self, num_iterations, max_actions, print_interval, batch_size):

        latest_score = deque(maxlen=50)
        episode = 0
        total_step = 0
        for epoh in range(num_iterations):
            state = self.env.reset()
            state = np.reshape(state, [1, 1])

            episode_reward = 0
            time_score = 0

            done = False

            bar = progressbar.ProgressBar(max_value=max_actions / 10,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()

            for step in range(max_actions):

                eps = self.get_eps(total_step)
                action = self.get_action(state, eps)

                next_state, reward, done, info = self.env.step(action)

                next_state = np.reshape(next_state, [1, 1])
                self.store(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                time_score += 1

                if done:
                    episode += 1
                    latest_score.append(episode_reward)
                    self.update_target_model()
                    ave_reward = np.mean(latest_score)
                    print("episode number: ", episode,", latest average reward: ",ave_reward , "time score: ", time_score)
                    self.record.write_summary(episode, episode_reward, latest_score, total_step,eps)
                    break

                if len(self.expirience_replay) > batch_size:
                    self.retrain(batch_size)

                if (total_step % 50 )==0 and step>0:
                    self.update_target_model()

                total_step += 1

            bar.finish()
            if (epoh + 1) % print_interval == 0:
                print("Episodes: {}".format(epoh + 1))
                self.env.render()









