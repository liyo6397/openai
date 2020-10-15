import tensorflow as tf
import datetime




def save_file(self, file_path = 'logs/gradient_tape/'):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = file_path+ current_time + '/train'
    writer = tf.summary.create_file_writer(log_dir)
