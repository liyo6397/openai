import tensorflow as tf


def initial_state(env):

    return tf.constant(env.reset(), dtype=tf.float32)

def convert_batchTensor(state):

    state = tf.constant(state, tf.float32)
    recent_state = tf.expand_dims(state, axis=0)

    return recent_state

def initial_policyVar():

    prob_action = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    return prob_action, values, rewards











