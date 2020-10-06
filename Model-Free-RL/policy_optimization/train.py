import tensorflow as tf
from Asyn_Learning import actor_critic
import numpy as np

class Worker:
    def __init__(self, name):
        self.name = 'w%02i' % name


    def train(self,params, epochs, local_steps_per_epoch):

        a3c = actor_critic(params.env, params.state)
        env = params.env


        for epoch in range(epochs):
            states = env.reset()
            states = np.reshape(states, [1, 1])
            for t in range(local_steps_per_epoch):
                if self.name == 'w00':
                    env.render()
                action = a3c.sample_DiscreteActions(states, hidden_sizes=(32,32))
