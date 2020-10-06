import gym
import tensorflow as tf
from DoubleDQN import Agent, doubleDQNetwork

if __name__ ==  '__main__':

    episodes = 100
    timesteps_per_episode = 10
    print_interval = 10
    batch_size = 32

    env = gym.make("Taxi-v3")
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
    agent = Agent(env, optimizer)

    QL = doubleDQNetwork(agent)

    QL.train(episodes, timesteps_per_episode, print_interval, batch_size)
    QL.make_video(env, agent)
