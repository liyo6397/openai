import tensorflow as tf
from collections import deque
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D,Lambda, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop

class Networks(Model):
    def __init__(self, env, agent_history_length):
        super().__init__()

        self.n_act = env.action_space.n
        self.normalize = Lambda(lambda x: x / 255.0)
        self.dense1 = Dense(units=32, activation='relu', input_shape=(1,))



        self.conv1 = Conv2D(filters=32, kernel_size=8, strides=4,
                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu",
                            input_shape=(None, 84, 84, agent_history_length))
        self.conv2 = Conv2D(filters=64, kernel_size=4, strides=2,
                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.conv3 = Conv2D(filters=64, kernel_size=3, strides=1,
                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation=None)
        self.flatten = Flatten()

    def forward(self, inputs):

        h0 = self.normalize(inputs)
        h1 = self.conv1(h0)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.flatten(h3)
        h5 = self.dense1(h4)

        return h5












    def network(self, inputs, hidden_sizes=(32,), activation = 'tanh', output_activation=None):

        layer = Dense(units=hidden_sizes[0], activation=activation, input_shape=(1,))(inputs)
        for h in self.hidden_sizes[1:-1]:
            layer = Dense(h, activation=activation)(layer)
        layer = Dense(self.hidden_sizes[-1], activation=output_activation)(layer)

        print("Output Size: ", hidden_sizes[-1])

        return layer

