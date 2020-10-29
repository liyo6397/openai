import unittest
import gym
import tensorflow as tf
from utils import Networks


class Test_util(unittest.TestCase):

    def setUp(self):

        self.env = gym.make("BreakoutNoFrameskip-v4")
        self.agent_history_length = 4
        self.net = Networks(self.env, self.agent_history_length)

    def test_inputsForCovN(self):

        print("Type of space: ",self.env.observation_space)


    def test_forward(self):

        state = self.env.reset()
        state = tf.constant(state, tf.float32)
        recent_state = tf.expand_dims(state, axis=0)


        logits = self.net.forward(recent_state)

        print(logits)

    def test_actor_critic(self):

        state = self.env.reset()
        state = tf.constant(state, tf.float32)
        recent_state = tf.expand_dims(state, axis=0)

        logits = self.net.forward(recent_state)

        pi = self.net.get_policy(logits)
        v = self.net.critic_value(logits)


        print("Actor: ",pi)
        print("Critic: ", v)



