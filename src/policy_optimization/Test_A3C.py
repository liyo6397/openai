import unittest
import gym
import tensorflow as tf
import utils
from actor_critic import A3C
from model import Networks
from train import trainer, Runner, A3C
from threading import Thread, Lock
from time import sleep
import tqdm
import numpy as np
import queue
class Test_par:
    def __init__(self):
        self.env_name = "BreakoutNoFrameskip-v4"
        self.gamma = 0.6
        self.num_episodes = 5
        self.max_steps_episode = 5
        self.learning_rate = 0.01
        self.agent_history_length = 4
        self.reward_threshold = 195
        self.num_process = 2
        self.betta = 0.01



class Test_util(unittest.TestCase):

    def setUp(self):

        self.env = gym.make("BreakoutNoFrameskip-v4")
        self.agent_history_length = 4
        self.n_act = self.env.action_space.n
        self.net = Networks(self.n_act, self.agent_history_length)
        self.par = Test_par()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.par.learning_rate)
        self.trainer = trainer(self.env, self.optimizer, self.par, i=0, lock=None)

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

        print("State shape: ", state.shape)



        print("Actor: ",logits_a)
        print("Critic: ", logits_c)

    def test_sample_action(self):

        for i in range(3):
            state = self.env.reset()
            state = tf.constant(state, tf.float32)

            state = utils.insert_axis0Tensor(state)

            logits_a, logits_c = self.net(state)


            action, prob = self.trainer.sample_action(logits_a)

            next_state, reward, done = self.trainer.tf_env_step(action)

        print(state.shape)

        print(logits_a)

        print(action)


    def test_entropy(self):
        state = self.env.reset()
        state = tf.constant(state, tf.float32)

        state = utils.insert_axis0Tensor(state)

        logits_a, logits_c = self.net(state)
        action, prob = self.trainer.sample_action(logits_a)
        entropy = self.trainer.produce_entropy(logits_a, prob)


        print(entropy)

    def test_raw_action(self):

        state = self.env.reset()
        state = convert_batchTensor(state)

        logits_a, logits_c = self.net(state)

        prob = tf.nn.softmax(logits_a)
        action = tf.random.categorical(prob, 1)
        print(action[0,0])

        print(action)


class Test_train(unittest.TestCase):

    def setUp(self):


        self.env = gym.make("BreakoutNoFrameskip-v4")
        self.env_name = "BreakoutNoFrameskip-v4"
        self.ini_state = self.env.reset()

        #Model
        self.par = Test_par()
        self.n_act = self.env.action_space.n
        self.model = Networks(self.n_act, agent_history_length=4)
        self.a3c = A3C(self.par.env_name, self.par)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.par.learning_rate)
        self.trainer = trainer(self.env, self.par)

    def test_get_expected_reward(self):

        prob_a, critic_val, rewards = self.trainer.explore(1)

        exp_rewards = self.a3c.get_expected_rewards(rewards, self.par.gamma)

        print(tf.math.reduce_mean(exp_rewards))

    def test_gym(self):

        env = gym.make(self.par.env_name)
        observation = env.reset()
        print(observation)
        for _ in range(1000):
            #env.render()
            action = env.action_space.sample()  # your agent here (this takes random actions)
            observation, reward, done, info = env.step(action)

            if done:
                observation = env.reset()
        env.close()

    def test_tf_env_step(self):

        state = tf.constant(self.env.reset(), dtype=tf.float32)
        state = utils.insert_axis0Tensor(state)
        logits_a, critic_val = self.model(state)

        action, prob_a = self.a3c.sample_action(logits_a)

        # Applying action to get next state and reward
        next_state, reward, done = self.trainer.tf_env_step(action)

        print(reward)

    def test_compute_loss(self):

        max_steps = 1000
        initial_state = tf.constant(self.env.reset(), dtype=tf.float32)
        prob_a, critic_val, rewards = self.trainer.explore(1, initial_state)

        exp_rewards = self.a3c.get_expected_rewards(rewards, self.par.gamma)

        loss = self.trainer.compute_loss(prob_a, critic_val, exp_rewards)

        print(loss)

    def test_run_episode(self):

        max_steps = 10
        with tqdm.trange(5) as episodes:

            for i in episodes:
                initial_state = tf.constant(self.env.reset(), dtype=tf.float32)


                rewards = self.trainer.run_episode(i, initial_state)


        #print("Probabilities: ", prob_a)
        #print("Critical values: ", critic_val)



    def test_main_training(self):

        #self.num_episodes = 1
        #self.max_steps_episode = 1000
        self.trainer.run()

    def test_multithread(self):

        threads = utils.create_threads(self.env, self.model, self.optimizer, self.a3c, self.par)

        process = []

        for thread in threads:
            thread.start()

            process.append(thread)



        # Wait for all threads to complete
        for t in process:
            t.join()

    def test_threadINtrainer(self):


        process = []
        envs = []

        for i in range(2):
            a3c_trainer = trainer(self.env, self.model, self.optimizer, self.par)
            a3c_trainer.start()
            process.append(a3c_trainer)

        for t in process:
            t.join()

    def test_threadTraining(self):

        process = []
        num_process = 2

        worker = Worker(num_process, self.par.env_name, self.par)
        worker.train()

    def test_globalModel(self):

        state = self.env.reset()
        state = tf.constant(state, tf.float32)

        state = utils.insert_axis0Tensor(state)
        model = self.model



        inputs = tf.random.normal(state.shape)
        print(inputs)
        model(inputs)

    def runner(self):

        #mem = utils.Memory()

       # with tqdm.trange(2) as episodes:
        episode = 0
        while True:
            mem = self.trainer.explore(episode, self.model)
            episode += 1
            yield mem


    def test_explore_queue(self):

        que = queue.Queue()

        mem = self.runner()


        for i in range(5):
            rec = next(mem)
            que.put(rec, timeout=100.0)
            #print("action:", rec.action)

        while not que.empty():
            mem = que.get()
            print(mem.prob_action)

    def get_queue(self, que):



        mem = que.get()
        while not que.empty():
            mem.concat(que.get())




        print("After concat: ",mem.prob_action)


    def test_get_queue(self):

        que = queue.Queue()

        mem = self.runner()


        for i in range(3):
            #que.put(var)
            #print("Put ", var)
            rec = next(mem)
            que.put(rec, timeout=100.0)
            print("Prob action:", rec.prob_action)

        self.get_queue(que)

    def test_runner(self):

        global_model = self.trainer.global_model
        worker = Runner(self.env, self.par, global_model)

        worker.run()

        que = worker.queue

        while not que.empty():
            mem = que.get()
            print(mem.prob_action)

class Test_A3Cclass(unittest.TestCase):

    def setUp(self):

        self.par = Test_par()
        self.env = gym.make(self.par.env_name)
        self.A3C = A3C(self.par.env_name, self.par)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.par.learning_rate)

    def test_runner(self):

        mem = self.A3C.get_queue()

        print(mem.prob_action)

    def test_expected_rewards(self):

        worker = Runner(self.env, self.par, self.A3C.global_model)
        worker.run()
        que = worker.queue
        mem = self.A3C.get_queue(que)

        a3c_trainer = trainer(self.env, self.optimizer, self.par, self.A3C.global_model)
        exp_rewards = a3c_trainer.get_expected_rewards(mem.rewards)

        print(exp_rewards)

    def test_loss(self):
        a3c_trainer = trainer(self.env, self.par)
        worker = Runner(self.env, self.par, self.A3C.global_model, a3c_trainer)
        worker.run()
        que = worker.queue
        mem = self.A3C.get_queue()


        exp_rewards = a3c_trainer.get_expected_rewards(mem.rewards)

        prob_a, c_values, exp_rewards, entropies = utils.insert_axis1Tensor(mem.prob_action, mem.critic_values,
                                                                            exp_rewards,
                                                                            mem.entropies)

        loss = a3c_trainer.compute_loss(prob_a, c_values, exp_rewards, entropies)

        print(loss)

    def test_grad_descent(self):



        episode_reward = self.A3C.grad_descent()

        print(episode_reward)














































