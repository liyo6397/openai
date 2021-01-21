import gym
import tqdm
import tensorflow as tf
from model import Networks
from train import trainer, A3C
import utils
import portpicker
import multiprocessing
import os
import json




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

    def __init__(self, num_worker=2, num_process=2):
        self.num_worker = num_worker
        self.num_process = num_process
        self.config = self.setup_config()




    def create_cluster(self, job_name):
        """Creates and starts local servers and returns the cluster_resolver."""
        worker_ports = [portpicker.pick_unused_port()for _ in range(self.num_worker)]
        #ps_ports = [portpicker.pick_unused_port() for _ in range(self.num_process)]

        #worker and parameter server need to know which port they need to listen to
        cluster = {}
        cluster[f'{job_name}'] = [f'host:{port}' for port in worker_ports]
        #cluster['process'] = [f'host:{port}' for port in ps_ports]

        return cluster

    def setup_config(self):

        worker_config = tf.compat.v1.ConfigProto()
        #if multiprocessing.cpu_count() < self.num_worker + 1:
        #    worker_config.inter_op_parallelism_threads = self.num_worker + 1
        worker_config.inter_op_parallelism_threads = 1

        return worker_config

    def create_server(self, cluster):

        cluster_spec = tf.train.ClusterSpec(cluster)

        for i in range(self.num_worker):
            worker_server = tf.distribute.Server(cluster_spec, job_name="worker", task_index=i, config=self.config, protocol="grpc")

        for i in range(self.num_process):
            ps_server = tf.distribute.Server(cluster_spec, job_name="process", task_index=i, protocol="grpc")

        cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
            cluster_spec, rpc_layer="grpc")
        return worker_server, ps_server, cluster_resolver

    def ParametersServerStrategy(self, cluster_resolver):

        #scale up model training on multiple machines
        #variable_partitioner = (
        #    tf.distribute.experimental.partitioners.FixedShardsPartitioner(
        #        num_shards=self.num_process))

        strategy = tf.distribute.experimental.ParameterServerStrategy(
            cluster_resolver)

        return strategy

    def setup_distributed(self, job_name):

        cluster_dict = self.create_cluster(job_name)
        os.environ["TF_CONFIG"] = json.dumps({'clusters': cluster_dict,
                                                         'task': {'type': 'worker', 'index': 0}})


@tf.function
def distributed_train_step(train_step, strategy):
    per_replica_losses = strategy.run(train_step)
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)
def check_point_dir():
    # Create a checkpoint directory to store the checkpoints.
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    return checkpoint_prefix

def run(num_episodes=10):

    clusters = cluster(2,2)

    clusters.setup_distributed(job_name='worker')

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    par = parameters()
    env = gym.make(par.env_name)


    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=par.learning_rate)
        model = Networks(env.action_space.n, agent_history_length=4)

        a3c = A3C(par, model, optimizer)

        a3c.start()


    for epoch in range(num_episodes):
        per_replica_loss = distributed_train_step(a3c.process, strategy)

        if epoch%2 == 0:
            print(f'Step {epoch}-Loss{per_replica_loss}')


