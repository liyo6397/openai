import tensorflow as tf
import os
import threading
from train import trainer
import multiprocessing
import gym


''''Variable Setting'''

def initial_state(env):

    return tf.constant(env.reset(), dtype=tf.float32)

def insert_axis0Tensor(state):

    state = tf.constant(state, tf.float32)
    recent_state = tf.expand_dims(state, axis=0)

    return recent_state

def initial_tensorArray():

    prob_action = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    entropy = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    terminal = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    return prob_action, values, rewards, entropy, terminal

def insert_axis1Tensor(*args):

    return [tf.expand_dims(x, 1) for x in args]

def nn_input_shape(env_game, atari=False):

    env = gym.make(env_game)
    state = env.reset()
    state = tf.constant(state, tf.float32)

    if atari:
        state = insert_axis0Tensor(state)

    return state.shape

'''File'''
class Writer:

    def __init__(self):

        self.loss_metric = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.score_metric = tf.keras.metrics.Mean('scores', dtype=tf.float32)
        self.writer = self.create_writer()

    def update_state(self, loss, rewards):

        # Record the loss and rewards for tensorboard
        self.loss_metric.update_state(loss)
        self.score_metric.update_state(rewards)

    def create_writer(self, log_dir='logs/'):

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        summary_dir = log_dir+'/summary'
        writer = tf.summary.create_file_writer(summary_dir)

        return writer

    def write_summary(self, episode):

        with self.writer.as_default():
            tf.summary.scalar("Episode", episode, step=episode)
            tf.summary.scalar("Loss", self.loss_metric.result(), step=episode)
            tf.summary.scalar("Average score", self.score_metric.result(), step=episode)

        self.loss_metric.reset_states()
        self.score_metric.reset_states()

class Memory():

    def __init__(self):

        self.prob_action = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.critic_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        self.entropies = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.terminal = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        self.states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.actions = []


    def experiance(self, state, action, reward, entropy):

        self.action = action
        self.reward = reward
        self.entropy = entropy
        self.state = state

        self.actions += [action]



    def training_data(self, t, prob_a, critic_val, done):
        self.prob_action = self.prob_action.write(t, prob_a[0, self.action])
        self.critic_values = self.critic_values.write(t, tf.squeeze(critic_val))
        self.rewards = self.rewards.write(t, self.reward)
        self.entropies = self.entropies.write(t, self.entropy)
        self.terminal = self.terminal.write(t, done)
        self.states = self.states.write(t, self.state)

    def to_stack(self):

        self.prob_action = self.prob_action.stack()
        self.critic_values = self.critic_values.stack()
        self.rewards = self.rewards.stack()
        self.entropies = self.entropies.stack()
        self.terminal = self.terminal.stack()
        self.states = self.states.stack()

    def concat(self, other):

        self.prob_action = tf.concat([self.prob_action, other.prob_action], 0)
        self.critic_values = tf.concat([self.critic_values, other.critic_values], 0)
        self.rewards = tf.concat([self.rewards, other.rewards], 0)
        self.entropies = tf.concat([self.entropies, other.entropies], 0)
        self.terminal = tf.concat([self.terminal, other.terminal], 0)

        self.actions.extend(other.actions)
        self.states.extend(other.states)






























