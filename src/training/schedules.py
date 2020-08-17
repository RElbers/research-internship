class UpdateSchedule:
    def __init__(self, per_iteration=True):
        self.per_iteration = per_iteration

    def __call__(self, info):
        return self.update(info.iteration)

    def update(self, idx):
        raise NotImplementedError()


class LearningRateScheduleWrapper(UpdateSchedule):
    def __init__(self, lr_scheduler, frequency=1):
        super().__init__()
        self.frequency = frequency
        self.scheduler = lr_scheduler
        self.accuracy = None

    def __call__(self, info):
        if info.iteration == 0:
            return

        if not info.iteration % self.frequency == 0:
            return

        self.accuracy = info.metrics_test()['accuracy']
        return self.update(info.iteration)

    def update(self, idx):
        return self.scheduler.step(self.accuracy)


class LinearDecay(UpdateSchedule):
    def __init__(self,
                 setter,
                 value_range,
                 time_range):
        super().__init__()
        self.setter = setter

        self.start_value = value_range[0]
        self.end_value = value_range[1]
        self.start_time = time_range[0]
        self.end_time = time_range[1]

        time_range = self.end_time - self.start_time
        value_range = self.end_value - self.start_value
        self.step_size = value_range / time_range

    def update(self, idx):
        if self.start_time <= idx <= self.end_time:
            n = (idx + 1) - self.start_time
            val = self.start_value + n * self.step_size
            self.setter(val)
