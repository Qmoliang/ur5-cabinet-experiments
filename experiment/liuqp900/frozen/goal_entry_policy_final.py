"""Activate by actual task stagnation; joint motion alone is not task progress."""
import numpy as np
from goal_entry_policy_v7 import LowDimPolicy as Previous

class LowDimPolicy(Previous):
    def __call__(self,c,s,arrays):
        if not self.escape_active:
            history=[item['error_m'] for item in self.history[-100:]]
            if len(history)==100 and history[0]-min(history)<.0002 and self.error>.002:
                self.escape_active=True;self.command_epoch+=1;self.brake_pending=np.max(np.abs(self.previous))>1e-8
                self.emit('actual_cartesian_stall_activates_target_end_module',error_m=self.error,q=self.q.tolist(),previous=self.previous.tolist(),nominal_joint_speed=float(np.max(np.abs(s['x']))),epoch=self.command_epoch)
        return super().__call__(c,s,arrays)
    def summary(self):
        out=super().summary();out.update(activation_requires_zero_joint_speed=False,activation_metric='actual Cartesian improvement of original LiuQP, not joint speed')
        return out
