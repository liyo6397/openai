import unittest
import gym
import tensorflow as tf
import utils
from actor_critic import A3C
from model import Networks
from train import trainer

class Test_par:
    def __init__(self):
        self.env_name = "BreakoutNoFrameskip-v4"
        self.gamma = 0.6
        self.num_episodes = 2
        self.max_steps_episode = 1000
        self.learning_rate = 0.01
        self.agent_history_length = 4
        self.reward_threshold = 195
        self.num_process = 2



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

        a3c = A3C(self.par)


        action, prob, log_prob = a3c.sample_action(logits_a)

        #print(logits_a)

        print(action)
        #print(prob)
        print(log_prob)

    def test_entropy(self):

        ini_state = initial_state(self.env)
        state = convert_batchTensor(ini_state)

        logits_a, logits_c = self.net(state)
        action, prob, log_prob = self.a3c.sample_action(logits_a)
        entropy = self.a3c.produce_entropy(logits_a, prob)


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
        self.a3c = A3C()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.par.learning_rate)
        #self.trainer = trainer(lambda : gym.make(self.env_name), self.model, self.optimizer, self.par)



    def test_run_episode(self):

        max_steps = 10
        prob_a, critic_val, rewards = self.trainer.explore(1)

        print("Probabilities: ", prob_a)
        print("Critical values: ", critic_val)
        print("Rewards: ", rewards)

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
        prob_a, critic_val, rewards = self.trainer.explore(1)

        exp_rewards = self.a3c.get_expected_rewards(rewards, self.par.gamma)

        loss = self.a3c.compute_loss(prob_a, critic_val, exp_rewards)

        print(loss)

    def test_train_episode(self):

        episode_reward = self.trainer.run_episode(1)

        print(episode_reward)

    def test_main_training(self):

        #self.num_episodes = 1
        #self.max_steps_episode = 1000
        self.trainer.train()

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

    def test_create_thread(self):

        #a3c_trainer = trainer(self.env, self.model, self.optimizer, self.par)

        utils.creat_process(self.env, self.optimizer, self.par, trainer)

























