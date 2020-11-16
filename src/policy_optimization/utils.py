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

    return prob_action, values, rewards, entropy

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

def create_writer(log_dir='logs/'):

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    summary_dir = log_dir+'/summary'
    writer = tf.summary.create_file_writer(summary_dir)

    return writer

def write_summary(writer, episode, score_metric, loss_metric):

    with writer.as_default():
        tf.summary.scalar("Episode", episode, step=episode)
        tf.summary.scalar("Loss", loss_metric.result(), step=episode)
        tf.summary.scalar("Average score", score_metric.result(), step=episode)






























