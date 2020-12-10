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

    # function: actor and critic network

    # function: actor policy
        # Add entropy into actor policy

    # function: replay experience buffer

    # function: update actor and critic netork by replay experience buffer

    # function calculating target values

    # function batch normalization

    # update target network by critic loss and actor policy
