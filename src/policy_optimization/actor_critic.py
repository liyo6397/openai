import numpy as np
import tensorflow as tf
import gym
import utils
from PIL import Image
from pyvirtualdisplay import Display



EPS = 1e-8


class A3C:



    def produce_entropy(self, logits, prob):


        #labels = [tf.one_hot(prob, depth=self.n_act)]
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=prob, logits=logits, axis=1)
        #entropy = tf.reduce_sum(tf.one_hot(pi, depth=self.n_act) * tf.nn.log_softmax(logits), axis=1)

        return entropy


    def sample_action(self, logits):

        action = tf.random.categorical(logits, 1)[0, 0]
        prob = tf.nn.softmax(logits)


        return action, prob

    def get_expected_rewards(self, rewards, gamma):

        # Transformed data into tensor shapes
        total_reward = tf.constant(0.0)
        total_reward_shape = total_reward.shape
        T = tf.shape(rewards)[0]
        exp_rewards = tf.TensorArray(dtype=tf.float32, size=T)

        #Accumalate the rewards from the end
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        for t in tf.range(T):
            total_reward = rewards[t]+gamma*total_reward
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
            exp_rewards: tf.Tensor) -> tf.Tensor:

        advantage = exp_rewards - critic_values
        log_prob = tf.math.log(action_probs)


        actor_loss = -tf.math.reduce_sum(log_prob*advantage)
        critic_loss = self.huber_loss(exp_rewards, critic_values)

        loss = actor_loss + critic_loss

        return loss

class visualization:

    def __init__(self, env: gym.Env):

        self.env = env
        self.setup_display()
        self.images = self.setup_images()

    def setup_display(self):
        display = Display(visible=0, size=(400, 300))
        display.start()

    def setup_images(self):

        screen = self.env.render(mode='rgb_array')

        # Creates an image memory from an object exporting the array interface
        im = Image.fromarray(screen)
        images = [im]

        return images



    def create_images(self, model: tf.keras.Model, a3c: 'A3C', max_steps_episodes):


        state = utils.initial_state(self.env)

        for i in range(max_steps_episodes):

            state = utils.insert_axis0Tensor(state)

            logits_a, critic_val = model(state)

            action, prob_a = a3c.sample_action(logits_a)
            state, reward, done, info = self.env.step(action)

            state = tf.constant(state, dtype=tf.float32)

            # Render screen every 10 steps
            if i % 10 == 0:
                screen = self.env.render(mode='rgb_array')
                self.images.append(Image.fromarray(screen))

            if done:
                break


    def save_image(self, image_file):
        self.images[0].save(
            image_file, save_all=True, append_images=self.images[1:], loop=0, duration=1)


























