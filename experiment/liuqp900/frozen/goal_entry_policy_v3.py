from pathlib import Path
import json,numpy as np
from goal_entry_policy_v2 import LowDimPolicy as Previous

class LowDimPolicy(Previous):
    def __init__(self,ctx,job,event):
        super().__init__(ctx,job,event)
        self.port=json.loads((Path(__file__).resolve().parent/'GOAL_BRANCH_PORTS.json').read_text())['ports'][job.get('port_rank',0)]
        self.qg=np.array(self.port['goal_q']);self.P=np.array(self.port['posture_matrix']);self.outward=np.array(self.port['outward']);self.radius=self.port['linear_radius_m'];self.goal_points=ctx.guard.positions(self.qg).copy()
    def _goal_gap(self):
        kind,i,j=self.blocker_id;guard=self.ctx.guard
        if kind in ('collision','frontier'):
            index=self.proxy_index[int(j)];distance,_=self.ctx.proxy_guard.point_geometry(self.goal_points[i][None,:],np.array([index]));return float(distance[0]-guard.radii[i])
        if kind=='self':return float(np.linalg.norm(self.goal_points[j]-self.goal_points[i])-guard.radii[i]-guard.radii[j])
        if kind=='workspace':
            axis=j%3;return float(self.goal_points[i,axis]-guard.workspace_lower[axis]-guard.radii[i] if j<3 else guard.workspace_upper[axis]-self.goal_points[i,axis]-guard.radii[i])
        raise ValueError(kind)
    def _extras(self,c,s):
        rows,bounds=super()._extras(c,s)
        if self.mode=='BYPASS':
            geometry=self._geometry()
            if geometry is not None:
                gap=geometry[3];goal_compatible_floor=min(c.near_distance,self._goal_gap());bounds[-1]=c.task_gain*(gap-goal_compatible_floor)
        return rows,bounds
    def summary(self):
        out=super().summary();out.update(posture_features='base-angle and elbow-branch coordinates from static terminal IK',bypass_clearance='minimum of original near distance and actual clearance at terminal posture; never lowers original physical safety')
        return out
