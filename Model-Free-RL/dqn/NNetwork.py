import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

class Networks():

    def __init__(self, n_act, units=24):
        #super().__init__()
        self.n_act = n_act



    def DeepNN_model(self, learning_rate=0.1,units=24):

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(units=units, activation='relu', input_shape=(1,)))
        model.add(Dense(units=units, activation='relu'))
        model.add(Dense(self.n_act, activation='linear'))

        return model


