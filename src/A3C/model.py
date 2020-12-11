from typing import Any, List, Sequence, Tuple
from collections import deque

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D,Lambda, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop

class Networks(Model):
    def __init__(self, num_actions: int, agent_history_length: int):
        super(Networks, self).__init__()

        self.n_act = num_actions
        self.normalize = Lambda(lambda x: x / 255.0)




        self.conv1 = Conv2D(filters=32, kernel_size=8, strides=4,
                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu",
                            input_shape=(None, 84, 84, agent_history_length))
        self.conv2 = Conv2D(filters=64, kernel_size=4, strides=2,
                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.conv3 = Conv2D(filters=64, kernel_size=3, strides=1,
                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation=None)
        self.flatten = Flatten()
        self.dense1 = Dense(units=512, activation='relu', input_shape=(1,))
        self.actor = Dense(units=self.n_act, activation=None)
        self.critic = Dense(units=1, activation=None)

    @tf.function
    def call(self, inputs: tf.Tensor)-> Tuple[tf.Tensor, tf.Tensor]:
        #h0 = self.normalize(inputs)
        h1 = self.conv1(inputs)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.flatten(h3)
        h5 = self.dense1(h4)
        actor_val = self.actor(h5)
        critic_val = self.critic(h5)

        return actor_val, critic_val
