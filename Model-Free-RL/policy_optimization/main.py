import gym
import tqdm
import tensorflow as tf
from model import Networks
from actor_critic import A3C
from train import trainer





class parameters:

    def __init__(self):
        self.env_name = "BreakoutNoFrameskip-v4"
        self.gamma = 0.99
        self.num_episodes = 10000
        self.max_steps_episode = 1000
        self.learning_rate = 0.01
        self.agent_history_length = 4
        self.reward_threshold = 195

if __name__ == "__main__":

    par = parameters()
    # gym environment
    env = gym.make(par.env_name)
    num_action = env.action_space.n
    max_steps_episode = par.max_steps_episode
    running_reward = 0
    episode = 0

    #class import
    model = Networks(num_action, agent_history_length=4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=par.learning_rate)
    a3c = A3C()
    a3c_trainer = trainer(env, model, optimizer, a3c, par)


    with tqdm.trange(par.num_episodes) as episodes:
        for t in episodes:
            episode_reward = int(a3c_trainer.train_episode(episode))

            running_reward = episode_reward * 0.01 + running_reward * .99

            t.set_description(f'Episode {episode}')
            t.set_postfix(
                episode_reward=episode_reward, running_reward=running_reward)

            # Show average episode reward every 10 episodes
            if episode % 10 == 0:
                pass  # print(f'Episode {i}: average reward: {avg_reward}')

            if running_reward > par.reward_threshold:
                break

    print(f'\nSolved at episode {episode}: average reward: {running_reward:.2f}!')
