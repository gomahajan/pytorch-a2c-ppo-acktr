class RolloutQueue(object):
    def __init__(self):
        self.queue = []
        self.current = 0

    def insert(self, rollouts):
        self.queue.append(rollouts)

    def pop(self):
        min = self.queue[0].returns.max()
        min_idx = 0
        for index, rollouts in enumerate(self.queue):
            if rollouts.returns.max() <= min:
                min_idx = index
                min = rollouts.returns.max()

        self.current = min_idx
        # if new_exp.value >= min:
        return self.queue[min_idx]

    def poppy(self):
        self.current = (self.current + 1) % len(self.queue)
        return self.queue[self.current]
