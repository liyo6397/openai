import tensorflow as tf
from Asyn_Learning import actor_critic
import numpy as np
from util import Buffer

class Worker:
    def __init__(self, name):
        self.name = 'w%02i' % name

    def update_reward(self, a3c, rewards, done):

        hidden_sizes = (32, 32)

        if done:
            total_reward = 0
        else:
            value = a3c.network(list(hidden_sizes) + [1])
            total_reward = value

        N = len(rewards)
        for i in range(N-2,-1):
            total_reward = rewards[i] + discount_factor*total_reward






    def train(self,params, epochs, max_timesteps):

        a3c = actor_critic(params.env, params.state)
        env = params.env
        buf = Buffer()



        for epoch in range(epochs):
            states = env.reset()
            states = np.reshape(states, [1, 1])

            ep_r = 0
            step = 0

            score = []
            while True or step < max_timesteps:
                if self.name == 'w00':
                    env.render()
                action = a3c.sample_DiscreteActions(states, hidden_sizes=(32,32))
                next_state, reward, done, info = env.step(action)

                buf.experiance_replay(action, reward, states, next_state)
                step += 1

                score.append(reward)
                if done:
                    break

            reward = self.update_reward(a3c, action, score, done, next_state)





