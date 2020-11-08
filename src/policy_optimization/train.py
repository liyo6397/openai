import tensorflow as tf
from actor_critic import A3C
import numpy as np
from typing import Any, List, Sequence, Tuple
import utils
import tqdm
import gym


class trainer():

    def __init__(self, env, model, optimizer, a3c, par):

        self.env = env()
        self.model = model
        self.optimizer = optimizer
        self.a3c = a3c
        self.par = par
        self.max_steps_episode = self.par.max_steps_episode
        self.num_episodes = self.par.num_episodes

        self.writer = utils.create_writer()
        self.loss_metric = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.score_metric = tf.keras.metrics.Mean('scores', dtype=tf.float32)

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

    def explore(self, episode: int) -> List[tf.Tensor]:

        prob_action, critic_values, rewards = utils.initial_policyVar()

        state = tf.constant(self.env.reset(), dtype=tf.float32)

        for t in tf.range(self.max_steps_episode):

            #Add outer barch axis for state
            state = utils.insert_axis0Tensor(state)

            # Run the model to get Q values for each action and critical values
            logits_a, critic_val = self.model(state)

            # Sampling action from its probability distribution
            action, prob_a = self.a3c.sample_action(logits_a)
            entropy = self.a3c.produce_entropy(prob_a, logits_a)

            # Applying action to get next state and reward
            next_state, reward, done = self.tf_env_step(action)
            state = next_state

            # Collect the trainning data
            prob_action = prob_action.write(t, prob_a[0, action])
            critic_values = critic_values.write(t, tf.squeeze(critic_val))
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                utils.write_summary(self.writer, episode, self.score_metric, self.loss_metric)
                self.loss_metric.reset_states()
                self.score_metric.reset_states()
                break

        prob_action = prob_action.stack()
        critic_values = critic_values.stack()
        rewards = rewards.stack()

        return prob_action, critic_values, rewards

    def run_episode(self, episode: int) -> tf.Tensor:

        with tf.GradientTape() as tape:

            prob_a, c_values, rewards = self.explore(episode)

            #Calculatr expected rewards for each time step
            exp_rewards = self.a3c.get_expected_rewards(rewards, self.par.gamma)

            # Convert training data to appropriate TF tensor shapes
            prob_a, c_values, exp_rewards = utils.insert_axis1Tensor(prob_a, c_values, exp_rewards)

            loss = self.a3c.compute_loss(prob_a, c_values, exp_rewards)

        grades = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grades, self.model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        self.loss_metric.update_state(loss)
        self.score_metric.update_state(rewards)

        return episode_reward

    def train(self):

        running_reward = 0

        with tqdm.trange(self.num_episodes) as episodes:
            for episode in episodes:
                episode_reward = int(self.run_episode(episode))

                running_reward = episode_reward * 0.01 + running_reward * .99

                episodes.set_description(f'Episode {episode}')
                episodes.set_postfix(
                    episode_reward=episode_reward, running_reward=running_reward)

                # Show average episode reward every 10 episodes
                if episode % 10 == 0:
                    pass  # print(f'Episode {i}: average reward: {avg_reward}')

                if running_reward > self.par.reward_threshold:
                    break

        print(f'\nSolved at episode {episode}: average reward: {running_reward:.2f}!')
























