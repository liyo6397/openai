import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

def DeepNN_model(n_act, units=24):
    inputs = tf.keras.Input(shape=(1,))

    outputs = Dense(units=units, activation='relu')(inputs)
    outputs = Dense(units=units, activation='relu')(outputs)
    outputs = Dense(n_act, activation='linear')(outputs)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam, loss='mse')

    return model