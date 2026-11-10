"""Goal-side support-plane deficits for colliding virtual continuation points.
The terminal feedback, observed map and physical safety certificates stay fixed.
"""
import numpy as np
from diagnostic_exact_pruned_policy import LowDimPolicy as Previous
class LowDimPolicy(Previous):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.goal_plane_cache={};self.goal_plane_rows=0;self.minimum_goal_plane_slack=float('inf')
 def plane(self,robot,obstacle):
  key=(int(robot),int(obstacle))
  if key not in self.goal_plane_cache:
   pg=self.ctx.proxy_guard;point=self.goal_points[key[0]];_,normal=pg.point_geometry(point[None,:],np.array([key[1]]));n=-normal[0];support=float(pg.support(n[None,:],np.array([key[1]]))[0]);offset=float(n@pg.centers[key[1]]+support+self.future_guard.radii[key[0]]+self.ctx.controller.safety_margin);slack=float(n@point-offset)
   assert slack>=-1e-8,('terminal_side_plane_incompatible',key,slack)
   self.minimum_goal_plane_slack=min(self.minimum_goal_plane_slack,slack);self.goal_plane_cache[key]=(n.copy(),offset)
  return self.goal_plane_cache[key]
 def continuation_witnesses(self,q):
  state=super().continuation_witnesses(q);rows=state['rows'];state['physical_sampled_deficit']=state['deficit']
  for fraction in sorted({r['fraction'] for r in rows if r['kind']=='environment'}):
   group=[r for r in rows if r['kind']=='environment' and r['fraction']==fraction];points=self.future_guard.positions((1-fraction)*q+fraction*self.qg).copy();jac=self._jac(points,[r['id'][0] for r in group])
   for r in group:
    i,j=r['id'];normal,offset=self.plane(i,j);r['physical_slack']=r['slack'];r['slack']=float(normal@points[i]-offset);r['b']=(1-fraction)*(normal@jac[i]);self.goal_plane_rows+=1
  state['deficit']=max([-r['slack'] for r in rows],default=0.);state['potential']=sum(r['slack']**2 for r in rows)/max(1,len(rows));return state
 def summary(self):
  out=super().summary();out.update(continuation_witness_objective='for currently colliding virtual points, use supporting planes facing the frozen terminal robot posture instead of the nearest local penetration normal; entry still requires the unchanged full continuous certificate',goal_side_planes=len(self.goal_plane_cache),goal_side_plane_rows=self.goal_plane_rows,minimum_terminal_support_slack_m=self.minimum_goal_plane_slack,optimized_future_configurations=0)
  return out
