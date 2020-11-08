import gym
import tqdm
import tensorflow as tf
from model import Networks
from actor_critic import A3C, visualization
from train import trainer
import utils
from utils import multiThread





class parameters:

    def __init__(self):
        self.env_name = "BreakoutNoFrameskip-v4"
        self.gamma = 0.99
        self.num_episodes = 10000
        self.max_steps_episode = 1000
        self.learning_rate = 0.01
        self.agent_history_length = 4
        self.reward_threshold = 195
        self.num_process = 3

if __name__ == "__main__":

    par = parameters()
    # gym environment
    env = gym.make(par.env_name)
    num_action = env.action_space.n
    max_steps_episode = par.max_steps_episode


    #class import
    model = Networks(num_action, agent_history_length=4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=par.learning_rate)
    a3c = A3C()
    trainer = trainer(env, model, optimizer, a3c, par)

    #set up threads
    threads = utils.create_threads(trainer, par.num_process)
    process = []

    for thread in threads:
        thread.start()
        process.append(thread)

    # Wait for all threads to complete
    for t in process:
        t.join()

    visual = visualization(env)
    visual.create_images(model, a3c, par.max_steps_episode)
    visual.save_image(image_file=f'{par.env_name}.gif')

