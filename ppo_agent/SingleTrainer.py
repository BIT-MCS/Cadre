class Trainer(object):
    def __init__(self, rollout, agent, trainer_cfg):
        self.rollout = rollout
        self.agent = agent
        self.trainer_cfg = trainer_cfg
