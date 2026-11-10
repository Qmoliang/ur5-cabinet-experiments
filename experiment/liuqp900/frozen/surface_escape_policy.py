"""Target-conditioned physical surface escape after continuation options stall."""
import numpy as np
from terminal_consistent_policy import LowDimPolicy as Previous
from diagnostic_exact_pruned_policy import LowDimPolicy as Geometry
from goal_entry_policy import clipped
class LowDimPolicy(Previous):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.surface_escape=False;self.surface_episodes=[];self.surface_nodes=[];self.surface_last_point=None;self.surface_travel=0.;self.surface_move_cycle=0;self.surface_last_travel=0.;self.surface_reversed=False;self.surface_hit_merit=None;self.surface_progress_best=np.inf;self.surface_pending=None;self.surface_visited=[]
 def _terms(self,c,s):
  if not self.surface_escape:return super()._terms(c,s)
  geom=self._geometry()
  if geom is None:return super()._terms(c,s)
  point,n,J,gap=geom;t=self._tangent(geom);K=np.vstack([n@J,t@J]);kind=self.blocker_id[0];margin=c.safety_margin if kind in ('collision','frontier') else self.ctx.guard.self_margin+4e-9 if kind=='self' else 0.
  desired=clipped(np.array([max(0.,c.task_gain*(margin-gap)),c.max_task_speed]),c.max_task_speed);inverse=np.linalg.pinv(K,rcond=1e-8);N=np.eye(6)-inverse@K;posture=self.P@N;z=self.P@(self.q-self.qg);prep=clipped(-c.task_gain*z-self.P@inverse@desired,c.max_task_speed)
  return c.task_weight*(K.T@K+posture.T@posture-s['J'].T@s['J']),c.task_weight*(s['J'].T@s['desired']-K.T@desired-posture.T@prep)
 def start_surface(self,c,s,arrays,raw):
  chosen=self._select_blocker(s,arrays,dict(velocity=s['x']))
  if chosen is not None:key=chosen[2];source='active_original_physical_row'
  else:
   env=[x for x in self.future_state['rows'] if x['kind']=='environment']
   if not env:return False
   w=min(env,key=lambda x:x['slack']);i,j=w['id'];key=('frontier' if self.ctx.proxy_guard.isbox[j] else 'collision',i,int(self.identifiers[j]));source='body_of_terminal_counterexample'
  self.blocker_id=key;self.surface_cache=None;geom=self._geometry()
  if geom is None:return False
  self.surface_escape=True;self.surface_hit_merit=float(np.sqrt(self.physical_merit(raw)));self.surface_progress_best=self.surface_hit_merit;self.surface_nodes=[];self.surface_travel=0.;self.surface_last_travel=0.;self.surface_move_cycle=self.k;self.surface_last_point=geom[0].copy();self.surface_reversed=False;self.circulation_tangent=None;self.reverse_used=False;self.tangent_override=None
  hit=self.feature(self.q);target=np.r_[self.ctx.target,self.P@(self.qg-self.memory_origin)];direction=target-hit;distance=float(np.linalg.norm(direction));direction/=max(distance,1e-12);delta=min(self.radius/2,distance/4)
  self.surface_pending=dict(normal=direction.tolist(),offset=float(direction@hit+delta),hit_feature=hit.tolist(),progress_buffer_m=delta,goal_value=distance-delta,hit_distance=distance,selected_row=list(key),hit_cycle=self.k)
  self.mode='BYPASS';self.command_epoch+=1;self.solve_cache={};self.brake_pending=np.max(np.abs(self.previous))>1e-8;self.actual_checkpoint=None
  record=dict(cycle=self.k,source=source,row=list(key),entry_merit=self.surface_hit_merit,q=self.q.tolist());self.surface_episodes.append(record);self.emit('physical_surface_escape_started',**{k:v for k,v in record.items() if k!='cycle'});return True
 def surface_call(self,c,s,arrays):
  self.expected=None;self.last_delta=np.zeros(6);self.current_s=s
  if self.brake_pending or self.terminal:return np.zeros(6)
  raw=Geometry.continuation_witnesses(self,self.q);M=float(np.sqrt(self.physical_merit(raw)))
  if not raw['rows'] and self.continuation_certificate():
   self.surface_escape=False;self.mode='CERT_ENTER';self.command_epoch+=1;self.brake_pending=np.max(np.abs(self.previous))>1e-8;self.actual_checkpoint=None;self.emit('surface_escape_opens_full_terminal_certificate');return np.zeros(6)
  geometry=self._geometry()
  if geometry is None:self.brake_pending=True;return np.zeros(6)
  point,n,J,gap=geometry
  if self.surface_last_point is not None:self.surface_travel+=float(np.linalg.norm(point-self.surface_last_point))
  self.surface_last_point=point.copy()
  if self.surface_travel-self.surface_last_travel>=.0002:self.surface_last_travel=self.surface_travel;self.surface_move_cycle=self.k
  f=self.feature(self.q);new_node=not self.surface_nodes or np.linalg.norm(f-np.array(self.surface_nodes[-1]['feature']))>=c.near_distance
  loop=False
  if new_node:
   loop=any(self.surface_travel-x['travel']>=2*c.near_distance and np.linalg.norm(f-np.array(x['feature']))<c.near_distance for x in self.surface_nodes[:-1]);self.surface_nodes.append(dict(feature=f.tolist(),travel=self.surface_travel,cycle=self.k))
  if (loop or self.k-self.surface_move_cycle>=100) and not self.surface_reversed and self.circulation_tangent is not None:
   self.circulation_tangent=-self.circulation_tangent;self.surface_reversed=True;self.command_epoch+=1;self.brake_pending=np.max(np.abs(self.previous))>1e-8;self.emit('surface_direction_reselected_from_actual_motion',reason='feature_revisit' if loop else 'no_real_boundary_motion',travel=self.surface_travel);return np.zeros(6)
  m=self.surface_pending;target=np.r_[self.ctx.target,self.P@(self.qg-self.memory_origin)];distance=float(np.linalg.norm(f-target));side=float(n@self._goal_normal(geometry));brake=self.previous-np.clip(self.previous,-self.acceleration*self.dt,self.acceleration*self.dt)
  ready=(m is not None and M<self.surface_hit_merit-.0002 and self._value(m,self.q)>=m['progress_buffer_m'] and distance<=m['hit_distance']-m['progress_buffer_m'] and side>=0)
  if ready and self.memory_safe(self.q,brake,m):
   committed=dict(m,activation_cycle=self.k);self.memories.append(committed);self.surface_visited.extend(self.surface_nodes);self.surface_pending=None;self.surface_escape=False;self.mode='CERT_PREP';self.actual_checkpoint=None;self.command_epoch+=1;self.brake_pending=np.max(np.abs(self.previous))>1e-8;self.option_best=M;self.option_last=self.k;self.emit('physical_surface_memory_committed',memory_index=len(self.memories)-1,actual_merit=M,hit_merit=self.surface_hit_merit,goal_side=side);return np.zeros(6)
  tangent=self._tangent(geometry);answer=self._try_direction(c,s,arrays,tangent@J)
  if answer is not None:
   self.circulation_tangent=tangent.copy()
   if self.k%100==0:self.emit('physical_surface_motion',merit=M,gap=gap,travel=self.surface_travel,side=side,safe_scale=answer[1][1])
   return self._publish(c,answer[0])
  # At a blocked tangent, posture or outward correction may open its cone.
  a=self._solve(c,s,arrays)
  if a is not None:
   ad=self._admit(c,arrays,a)
   if ad is not None and float((self.P@(self.q-self.qg))@(self.P@ad[0]))<-1e-8:return self._publish(c,a)
  b=n@J;b/=max(np.linalg.norm(b),1e-12);answer=self._try_direction(c,s,arrays,b)
  if answer is not None:return self._publish(c,answer[0])
  self.brake_pending=True;return np.zeros(6)
 def __call__(self,c,s,arrays):
  if self.surface_escape:return self.surface_call(c,s,arrays)
  delta=super().__call__(c,s,arrays)
  if self.escape_active and self.mode=='CERT_PREP' and not self.brake_pending and self.option_last is not None and self.k-self.option_last>=100 and len(self.option_tried)>=len(self.terminal_options):
   raw=Geometry.continuation_witnesses(self,self.q)
   if self.start_surface(c,s,arrays,raw):self.expected=None;c.pilot_supplied_solution=None;return np.zeros(6)
  return delta
 def summary(self):
  out=super().summary();out.update(t0_revision='physical_surface_after_terminal_exhaustion',physical_surface_episodes=self.surface_episodes,physical_surface_memory_nodes=self.surface_visited+self.surface_nodes,physical_surface_active=self.surface_escape,surface_control_dimension=2,posture_control_dimension=2,scene_specific_direction=False);return out
