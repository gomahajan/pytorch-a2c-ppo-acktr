class RolloutQueue(object):
    def __init__(self):
        self.queue = []

    def insert(self, rollouts):
        self.queue.append(rollouts)

    def pop(self):
        min = self.queue[0].returns.max()
        min_idx = 0
        for index, rollouts in enumerate(self.queue):
            if rollouts.returns.max() <= min:
                min_idx = index
                min = rollouts.returns.max()

        # if new_exp.value >= min:
        return self.queue[min_idx]