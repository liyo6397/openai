import tensorflow as tf
from Asyn_Learning import actor_critic
import numpy as np
from util import Buffer

class Worker:
    def __init__(self, name):
        self.name = 'w%02i' % name

    def update_reward(self, a3c, action, reward, done, next_state):

        hidden_sizes = (32, 32)

        if done:
            reward = 0
        else:
            value = a3c.network(list(hidden_sizes) + [1])
            reward = value





    def train(self,params, epochs, steps_per_epoch):

        a3c = actor_critic(params.env, params.state)
        env = params.env
        buf = Buffer()



        for epoch in range(epochs):
            states = env.reset()
            states = np.reshape(states, [1, 1])

            ep_r = 0
            step = 0
            while True or step < steps_per_epoch:
                if self.name == 'w00':
                    env.render()
                action = a3c.sample_DiscreteActions(states, hidden_sizes=(32,32))
                next_state, reward, done, info = env.step(action)

                buf.experiance_replay(action, reward, states, next_state)
                step += 1

                reward = self.update_reward(a3c, action, reward, done, next_state)





