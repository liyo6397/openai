import numpy as np
import tensorflow as tf
import gym
import random
from collections import deque
import os

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop
from NNetwork import Networks


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
        self.learning_rate = 0.01
        self.net = Networks(self.n_act)
        self.q_network = self.net.DeepNN_model()
        self.q_target_network = self.net.DeepNN_model()
        self.loss = tf.keras.losses.Huber()
        self.loss_metric = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.q_metric = tf.keras.metrics.Mean('Q_values', dtype=tf.float32)

        #Summary
        self.writer = self.create_writer()
        #self.record = summary()



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




    def get_action(self, state, eps=0.1):

        if random.uniform(0, 1) <= eps:
            return self.env.action_space.sample()  # Explore action space

        q_values = self.q_network.predict(state)  # exploit action space
        action = np.argmax(q_values[0])

        return action
    def retrain_modelFit(self, batch_size):

        minibatch = random.sample(self.expirience_replay, batch_size)


        for state, action, reward, done, next_state in minibatch:

            target = self.q_network.predict(state)


            if done:
                target[0][action] = reward
            else:
                tar = self.q_target_network.predict(next_state)
                target[0][action] = reward + self.discount_factor*np.amax(tar)





            self.q_network.fit(state, target, verbose= 0)

    def retrain_gradientTape(self, batch_size):

        minibatch = random.sample(self.expirience_replay, batch_size)


        for state, action, reward, done, next_state in minibatch:
            with tf.GradientTape() as tape:

                q_model = self.q_network
                tf_curr_q = q_model(state)
                curr_q = q_model.predict(state)
                expected_q = curr_q

                if done:
                    expected_q[0][action] = reward
                else:
                    target_model = self.q_target_network
                    tar = target_model.predict(next_state)
                    expected_q[0][action] = reward + self.discount_factor*np.amax(tar)

                tf_expected_q = tf.constant(expected_q)

                loss_values = self.loss(tf_curr_q, tf_expected_q)
            grades = tape.gradient(loss_values, self.q_network.trainable_variables)
            self.optimizer.apply_gradients(zip(grades, self.q_network.trainable_variables))

            self.loss_metric.update_state(loss_values)
            self.q_metric.update_state(curr_q)


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

                if done or step == max_actions -1:
                    episode += 1
                    latest_score.append(episode_reward)
                    self.update_target_model()
                    ave_reward = np.mean(latest_score)
                    ave_q = self.q_metric.result()
                    print("episode number: ", episode,", Average reward: ",ave_reward , "Average Q: ", ave_q)
                    self.write_summary(episode, latest_score, episode_reward, total_step, eps)

                    break

                if len(self.expirience_replay) > batch_size:
                    self.retrain_gradientTape(batch_size)

                if (total_step % 50 )==0 and step>0:
                    self.update_target_model()

                total_step += 1

            bar.finish()
            if (epoh + 1) % print_interval == 0:
                print("Episodes: {}".format(epoh + 1))
                self.env.render()


    def create_writer(self, log_dir='logs/'):

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        summary_dir = log_dir+'/summary'
        writer = tf.summary.create_file_writer(summary_dir)

        return writer

    def write_summary(self, episode, latest_100_score, episode_score, total_step, eps):

        with self.writer.as_default():
            tf.summary.scalar("Reward (clipped)", episode_score, step=episode)
            tf.summary.scalar("Latest 100 avg reward (clipped)", np.mean(latest_100_score), step=episode)
            tf.summary.scalar("Loss", self.loss_metric.result(), step=episode)
            tf.summary.scalar("Average Q", self.q_metric.result(), step=episode)
            tf.summary.scalar("Total Frames", total_step, step=episode)
            tf.summary.scalar("Epsilon", eps, step=episode)

        self.loss_metric.reset_states()
        self.q_metric.reset_states()





