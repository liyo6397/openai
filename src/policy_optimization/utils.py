import tensorflow as tf
import os
import threading
from train import trainer

'''Thread'''

class multiThread(threading.Thread):

    def __init__(self, trainer: 'trainer'):
        super().__init__()

        self.trainer = trainer
        self.threadLock = threading.Lock()

    def run(self):

        self.threadLock.acquire()
        self.trainer.train()
        self.threadLock.release()

def create_threads(target, num_process):

    threads = []
    for i in range(num_process):
        thread = multiThread(target)
        threads.append(thread)

    return threads

''''Variable Setting'''

def initial_state(env):

    return tf.constant(env.reset(), dtype=tf.float32)

def insert_axis0Tensor(state):

    state = tf.constant(state, tf.float32)
    recent_state = tf.expand_dims(state, axis=0)

    return recent_state

def initial_policyVar():

    prob_action = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    return prob_action, values, rewards

def insert_axis1Tensor(*args):

    return [tf.expand_dims(x, 1) for x in args]

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






























