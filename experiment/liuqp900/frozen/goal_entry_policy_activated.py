"""Initialize escape-progress references at the actual controller handoff."""
import numpy as np
from goal_entry_policy_final import LowDimPolicy as Previous

class LowDimPolicy(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.activation_progress_resets=0
    def __call__(self,c,s,arrays):
        if not self.escape_active:
            history=[item['error_m'] for item in self.history[-100:]]
            if len(history)==100 and history[0]-min(history)<.0002 and self.error>.002:
                assert not self.memories and self.pending_memory is None
                self.feature_best=float(np.linalg.norm(self.feature(self.q)-self.goal_feature));self.feature_last_progress=self.k;self.feature_mode=self.mode
                self.full_posture_best=float(np.sqrt(self.error**2+self.posture_lever**2*np.sum((self.q-self.qg)**2)));self.full_posture_last_progress=self.k;self.full_posture_mode=self.mode
                self.best_executed_error=self.error;self.last_improvement_cycle=self.k;self.progress_mode=self.mode
                self.no_action_start=None;self.solve_cache={};self.activation_progress_resets+=1
                self.escape_active=True;self.command_epoch+=1;self.brake_pending=np.max(np.abs(self.previous))>1e-8
                self.emit('actual_cartesian_stall_activates_target_end_module',error_m=self.error,q=self.q.tolist(),previous=self.previous.tolist(),nominal_joint_speed=float(np.max(np.abs(s['x']))),epoch=self.command_epoch,progress_history_reset=True)
        return super().__call__(c,s,arrays)
    def summary(self):
        out=super().summary();out.update(activation_progress_resets=self.activation_progress_resets,pre_activation_progress_not_used_for_escape_decisions=True)
        return out
