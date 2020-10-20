import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

class Networks():

    def __init__(self, n_act, units=24):
        #super().__init__()
        self.n_act = n_act

    def DeepNN_model(self, units=24):
        inputs = tf.keras.Input(shape=(1,))

        outputs = Dense(units=units, activation='relu')(inputs)
        outputs = Dense(units=units, activation='relu')(outputs)
        outputs = Dense(self.n_act, activation='linear')(outputs)

        model = Model(inputs, outputs)
        #model.compile(optimizer=Adam, loss='mse')

        return model


