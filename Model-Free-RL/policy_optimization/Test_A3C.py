import unittest
import gym
import tensorflow as tf
from utils import Networks, convert_batchTensor, initial_state
from actor_critic import A3C


class Test_util(unittest.TestCase):

    def setUp(self):

        self.env = gym.make("BreakoutNoFrameskip-v4")
        self.agent_history_length = 4
        self.n_act = self.env.action_space.n
        self.net = Networks(self.n_act, self.agent_history_length)
        self.a3c = A3C()

    def test_inputsForCovN(self):

        print("Type of space: ",self.env.observation_space)


    def test_forward(self):

        state = self.env.reset()
        state = tf.constant(state, tf.float32)
        recent_state = tf.expand_dims(state, axis=0)

        logits =self.net(recent_state)


        #logits = self.net.forward(recent_state)

        print(logits)

    def test_actor_critic(self):

        state = self.env.reset()
        state = tf.constant(state, tf.float32)
        recent_state = tf.expand_dims(state, axis=0)

        logits_a, logits_c = self.net(recent_state)



        print("Actor: ",logits_a)
        print("Critic: ", logits_c)

    def test_sample_action(self):

        state = self.env.reset()
        state = convert_batchTensor(state)

        logits_a, logits_c = self.net(state)

        a3c = A3C()

        action = a3c.sample_action(logits_a)

        print(logits_a)

        print(action)

    def test_entropy(self):

        ini_state = initial_state(self.env)
        state = convert_batchTensor(ini_state)

        logits_a, logits_c = self.net(state)

        entropy = self.a3c.produce_entropy(logits_a)
        action = self.a3c.sample_action(logits_a)

        print(entropy)










