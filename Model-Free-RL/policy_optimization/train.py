import tensorflow as tf
from actor_critic import A3C
import numpy as np
from typing import Any, List, Sequence, Tuple

def explore_episode(initial_state: tf.Tensor,
    model: tf.keras.Model,
    max_steps: int) -> List[tf.Tensor]:







