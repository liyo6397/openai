import tensorflow as tf
import numpy as np
from typing import Any, List, Sequence, Tuple
import utils
import tqdm
import threading
from threading import Lock, Thread
from model import Networks
import gym
from time import sleep
import queue


class trainer:
#class trainer():

    def __init__(self, env, par, i=0, lock=None):


        self.env = env


        self.par = par
        self.max_steps_episode = self.par.max_steps_episode
        self.num_episodes = self.par.num_episodes


        self.writer = utils.Writer()
        self.loss_metric = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.score_metric = tf.keras.metrics.Mean('scores', dtype=tf.float32)
        self.lock = lock


    # Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
    # This would allow it to be included in a callable TensorFlow graph.

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:

        def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Returns state, reward and done flag given an action."""

            state, reward, done, _ = self.env.step(action)
            return (state.astype(np.float32),
                    np.array(reward, np.int32),
                    np.array(done, np.int32))

        return tf.numpy_function(env_step, [action],
                                   [tf.float32, tf.int32, tf.int32])

    def produce_entropy(self, logits, prob):


        #labels = [tf.one_hot(prob, depth=self.n_act)]
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=prob, logits=logits)
        #entropy = tf.reduce_sum(tf.one_hot(pi, depth=self.n_act) * tf.nn.log_softmax(logits), axis=1)
        #entropy = tf.reduce_sum(prob*tf.math.log(prob))

        return entropy


    def sample_action(self, logits):

        action = tf.random.categorical(logits, 1)[0,0]
        prob = tf.nn.softmax(logits)


        return action, prob

    def get_expected_rewards(self, rewards):

        # Transformed data into tensor shapes
        total_reward = tf.constant(0.0)
        total_reward_shape = total_reward.shape
        T = tf.shape(rewards)[0]
        exp_rewards = tf.TensorArray(dtype=tf.float32, size=T)

        #Accumalate the rewards from the end
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        for t in tf.range(T):
            total_reward = rewards[t]+self.par.gamma*total_reward
            total_reward.set_shape(total_reward_shape)
            exp_rewards = exp_rewards.write(t, total_reward)
        exp_rewards = exp_rewards.stack()[::-1]

        return exp_rewards

    def huber_loss(self, rewards, values):

        huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        return huber(rewards, values)

    def compute_loss(self,
            action_probs: tf.Tensor,
            critic_values: tf.Tensor,
            exp_rewards: tf.Tensor,
            entropies: tf.Tensor) -> tf.Tensor:

        advantage = exp_rewards - critic_values
        log_prob = tf.math.log(action_probs)


        actor_loss = -tf.math.reduce_sum(log_prob*advantage)
        critic_loss = self.huber_loss(exp_rewards, critic_values)-self.par.betta*entropies

        loss = actor_loss + critic_loss

        return loss


    def explore(self, episode: int, MLmodel: tf.keras.Model): #-> List[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Memory]:

        state = tf.constant(self.env.reset(), dtype=tf.float32)


        mem = utils.Memory()

        for t in tf.range(self.max_steps_episode):

            #Add outer barch axis for state

            state = tf.expand_dims(state, 0)

            # Run the model to get Q values for each action and critical values
            logits_a, critic_val = MLmodel(state)

            # Sampling action from its probability distribution
            action, prob_a = self.sample_action(logits_a)
            entropy = self.produce_entropy(prob_a, logits_a)

            # Applying action to get next state and reward

            next_state, reward, done = self.tf_env_step(action)

            state = next_state

            mem.experiance(state, action, reward, entropy)
            # Collect the trainning data

            mem.training_data(t, prob_a, critic_val, done)


            if tf.cast(done, tf.bool):
                utils.write_summary(self.writer, episode, self.score_metric, self.loss_metric)
                self.loss_metric.reset_states()
                self.score_metric.reset_states()
                break


        mem.to_stack()

        return mem

    def grad_descent(self, que_data):


        with tf.GradientTape() as tape:
            # Collect data from runner
            #mem = self.get_queue()
            # Calculatr expected rewards for each time step
            exp_rewards = self.get_expected_rewards(que_data.rewards)

            # Convert training data to appropriate TF tensor shapes
            prob_a, c_values, exp_rewards, entropies = utils.insert_axis1Tensor(que_data.prob_action, que_data.critic_values,
                                                                                exp_rewards,
                                                                                que_data.entropies)

            loss = self.compute_loss(prob_a, c_values, exp_rewards, entropies)


        grades = tape.gradient(loss, self.local_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grades, self.global_model.trainable_variables))
        self.local_model.set_weights(self.global_model.get_weights())

        episode_reward = tf.math.reduce_sum(que_data.rewards)

        return loss, que_data.rewards, episode_reward

    def run_episode(self, episode: int, initial_state: tf.Tensor) -> [tf.Tensor]:

        with tf.GradientTape() as tape:

            prob_a, c_values, rewards, entropies, memory = self.explore(episode, initial_state)

            #Calculatr expected rewards for each time step
            exp_rewards = self.get_expected_rewards(rewards)

            # Convert training data to appropriate TF tensor shapes
            prob_a, c_values, exp_rewards, entropies = utils.insert_axis1Tensor(prob_a, c_values, exp_rewards, entropies)

            loss = self.compute_loss(prob_a, c_values, exp_rewards, entropies)

        grades = tape.gradient(loss, self.local_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grades, self.global_model.trainable_variables))

        self.local_model.set_weights(self.global_model.get_weights())

        episode_reward = tf.math.reduce_sum(rewards)

        self.loss_metric.update_state(loss)
        self.score_metric.update_state(rewards)

        return episode_reward




def tf_env_step(env, action: tf.Tensor) -> List[tf.Tensor]:

    def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""

        state, reward, done, _ = env.step(action)
        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32))

    return tf.numpy_function(env_step, [action],
                           [tf.float32, tf.int32, tf.int32])



def policy_runner(env, model, max_steps_episode=5):

    state = tf.constant(env.reset(), dtype=tf.float32)

    initial_state_shape = state.shape

    mem = utils.Memory()
    for t in tf.range(max_steps_episode):

        # Add outer barch axis for state

        state = tf.expand_dims(state, 0)

        # Run the model to get Q values for each action and critical values
        logits_a, critic_val = model(state)

        # Sampling action from its probability distribution
        action = tf.random.categorical(logits_a, 1)[0,0]

        # Applying action to get next state and reward
        next_state, reward, done = tf_env_step(env, action)

        #state.set_shape(initial_state_shape)
        state = next_state

        mem.experiance(t, state, action, reward, done)

    mem.to_stack()

    return mem
class Runner(Thread):

    def __init__(self, env, par, model):
        super().__init__()
        self.queue = queue.Queue()
        self.num_process = par.num_process
        self.env = env
        self.par = par
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.par.learning_rate)
        self.model = model

        self.queue = queue.Queue()
        self.num_episodes = self.par.num_episodes
        self.max_steps_episode = self.par.max_steps_episode
        self.trainer = trainer


    def start_runner(self, threadID):
        print("Thread: ", threadID)
        self.start()

    def collect_data(self):


        episode = 0
        while True:
            #mem = self.trainer.explore(episode, self.model)
            mem = policy_runner(self.env, self.model)
            episode += 1
            yield mem


    def run(self):

        mem = self.collect_data()
        for i in range(self.max_steps_episode):
            self.queue.put(next(mem), timeout=10.0)
            self.queue.task_done()

    def train(self):



        process = []
        lock = Lock()


        for i in range(self.num_process):
            env = gym.make(self.game_name)
            # Set seed for experiment reproducibility
            self.set_seed(env, seed=42)
            process.append(trainer(env, self.optimizer, self.par, self.global_model, i, lock))


        for i, worker in enumerate(process):
            worker.start()

        [w.join() for w in process]

class A3C:

    def __init__(self, game_name, par):

        self.par = par
        self.env = gym.make(game_name)
        self.set_seed(self.env)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.par.learning_rate)

        #set up model
        self.global_model = Networks(self.env.action_space.n, agent_history_length=4)
        inputs_shape = utils.nn_input_shape(game_name, atari=True)
        self.inputs = tf.random.normal(inputs_shape)
        logits, c_val = self.global_model(self.inputs)
        self.local_model = Networks(self.env.action_space.n, agent_history_length=4)



        #set up workers
        self.runner = Runner(self.env, par, self.local_model)

        # set up local model
        self.rewards = tf.constant([0., 0., 0., 0., 0.])
        #grads = self.grad_descent(self.inputs, self.rewards)
        # copy weights from the parameter server to the local model
        #self.sync_weights(grads)

        # For recording results
        self.loss_metric = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.score_metric = tf.keras.metrics.Mean('scores', dtype=tf.float32)

    def sample_action(self, logits):

        action = tf.random.categorical(logits, 1)[:,0]
        prob = tf.nn.softmax(logits)


        return action, prob

    def setup_localmodel(self, states):



        # Run the model to get Q values for each action and critical values
        logits_a, critic_val = self.local_model(states)
        # Sampling action from its probability distribution
        action, prob_a = self.sample_action(logits_a)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=prob_a, logits=logits_a)
        critic_val = tf.squeeze(critic_val)


        action = action.numpy()
        prob_a = prob_a.numpy()
        prob_a = [prob_a[i, act] for i, act in enumerate(action)]
        prob_a = tf.convert_to_tensor(prob_a, dtype=tf.float32)

        critic_val = tf.convert_to_tensor(critic_val, dtype=tf.float32)
        entropy = tf.convert_to_tensor(entropy, dtype=tf.float32)

        return logits_a, prob_a, action, critic_val, entropy

    def get_expected_rewards(self, rewards):

        # Transformed data into tensor shapes
        total_reward = tf.constant(0.0)
        total_reward_shape = total_reward.shape
        T = tf.shape(rewards)[0]
        exp_rewards = tf.TensorArray(dtype=tf.float32, size=T)

        # Accumalate the rewards from the end
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        for t in tf.range(T):
            total_reward = rewards[t] + self.par.gamma * total_reward
            total_reward.set_shape(total_reward_shape)
            exp_rewards = exp_rewards.write(t, total_reward)
        exp_rewards = exp_rewards.stack()[::-1]

        return exp_rewards

    def huber_loss(self, rewards, values):

        huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        return huber(rewards, values)

    def compute_loss(self, prob_a, c_val, exp_rewards, entropies):


        # Convert training data to appropriate TF tensor shapes
        prob_a, critic_values, exp_rewards, entropies = utils.insert_axis1Tensor(prob_a,
                                                                                  exp_rewards,
                                                                            c_val,
                                                                            entropies)

        advantage = exp_rewards - critic_values
        log_prob = tf.math.log(prob_a)

        actor_loss = -tf.math.reduce_sum(log_prob * advantage)
        critic_loss = self.huber_loss(exp_rewards, critic_values) - self.par.betta * entropies

        loss = actor_loss + critic_loss

        return loss


    def set_seed(self, env, seed=42):
        env.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

    def start(self, threadID):
        self.runner.start_runner(threadID)

    def get_queue(self):

        #self.runner.run()
        que = self.runner.queue
        que_data = que.get()
        while not que.empty():

            que_data.concat(que.get())

        return que_data

    def grad_descent(self, states, rewards):

        with tf.GradientTape() as tape:

            logits_a, prob_a, action, c_val, entropies = self.setup_localmodel(states)

            # Calculatr expected rewards for each time step
            exp_rewards = self.get_expected_rewards(rewards)


            # Convert training data to appropriate TF tensor shapes
            prob_a, c_values, exp_rewards, entropies = utils.insert_axis1Tensor(prob_a,
                                                                                c_val,
                                                                                exp_rewards,
                                                                                entropies)

            loss = self.compute_loss(prob_a, c_values, exp_rewards, entropies)

        grads = tape.gradient(loss, self.local_model.trainable_variables)
        episode_reward = tf.math.reduce_sum(rewards)



        return grads, episode_reward

    def sync_weights(self, grads):
        # Sync local model weights with global model
        self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_variables))
        self.local_model.set_weights(self.global_model.get_weights())
































