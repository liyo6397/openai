from collections import deque

class Buffer:
    def __init__(self):

        self.record = deque()

    def experiance_replay(self, action, reward, states, next_state):

        self.record.append((states, next_state, reward, action))