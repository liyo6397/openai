import tensorflow as tf
import datetime
import os
import numpy as np

class summary:

    def __init__(self):

        self.writer = self.create_writer()

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
