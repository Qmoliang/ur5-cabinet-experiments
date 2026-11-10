"""Clear per-solve expected-command state also on early braking returns."""
import numpy as np
from blocker_policy import LowDimPolicy as Previous

class LowDimPolicy(Previous):
    def __call__(self,c,s,arrays):
        self.expected=None;self.last_delta=np.zeros(6)
        if self.error<=1e-3 and self.mode=='OBSTACLE_RELEASE':
            self.switch('RESERVE_GOAL','goal_tolerance_reached');self.blocker_id=None
        return super().__call__(c,s,arrays)
