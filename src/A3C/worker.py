import gym
import tqdm
import tensorflow as tf
from model import Networks
from actor_critic import A3C, visualization
from train import trainer
import utils
import portpicker
import multiprocessing




class parameters:

    def __init__(self):
        self.env_name = "BreakoutNoFrameskip-v4"
        self.num_episodes = 10000
        self.max_steps_episode = 1000

        self.learning_rate = 0.01
        self.agent_history_length = 4
        self.reward_threshold = 195
        self.num_process = 3

        self.gamma = 0.99 # discount factor for accumulating reward
        self.betta = 0.01 # strength of the entropy

class cluster:

    def __init__(self, num_worker, num_process):
        self.num_worker = num_worker
        self.num_process = num_process
        self.config = self.setup_config()



    def create_cluster(self):
        """Creates and starts local servers and returns the cluster_resolver."""
        worker_ports = [portpicker.pick_unused_port()for _ in range(self.num_worker)]
        ps_ports = [portpicker.pick_unused_port() for _ in range(self.num_process)]

        #worker and parameter server need to know which port they need to listen to
        cluster = {}
        cluster['worker'] = [f'host:{port}' for port in worker_ports]
        cluster['process'] = [f'host:{port}' for port in ps_ports]

        return cluster

    def setup_config(self):

        worker_config = tf.compat.v1.ConfigProto()
        #if multiprocessing.cpu_count() < self.num_worker + 1:
        #    worker_config.inter_op_parallelism_threads = self.num_worker + 1
        worker_config.inter_op_parallelism_threads = 1

        return worker_config

    def run(self, cluster):

        cluster_spec = tf.train.ClusterSpec(cluster)

        for i in range(self.num_worker):
            tf.distribute.Server(cluster_spec, job_name="worker", task_index=i, config=self.config, protocol="grpc")

        for i in range(self.num_process):
            tf.distribute.Server(cluster_spec, job_name="process", task_index=i, protocol="grpc")









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
    trainer = trainer(env, model, optimizer, par)

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

