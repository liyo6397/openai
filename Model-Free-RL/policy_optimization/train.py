import tensorflow as tf
#from actor_critic import A3C
import numpy as np
from typing import Any, List, Sequence, Tuple
from utils import initial_policyVar, convert_batchTensor
from model import Networks

def explore_episode(initial_state: tf.Tensor,
    model: tf.keras.Model,
    a3c: 'A3C',
    max_steps: int) -> List[tf.Tensor]:

    prob_action, critic_values, rewards = initial_policyVar()

    for t in tf.range(max_steps):

        state = convert_batchTensor(initial_state)

        logits_a, critic_val = model(state)

        action, prob_action = a3c.sample_action()
        entropy = a3c.produce_entropy(prob_action, logits_a)

        critic_values = critic_values.write(t, tf.squeeze(critic_val))









