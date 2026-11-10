"""Finite terminal-pose alternatives, scalar continuation assessment, no route search."""
import json,numpy as np
from pathlib import Path
from posture_policy import LowDimPolicy as Previous
from diagnostic_exact_pruned_policy import LowDimPolicy as Geometry
import pilot_adapter
class LowDimPolicy(Previous):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw)
  self.memory_origin=self.qg.copy();self.terminal_options=json.loads((pilot_adapter.ROOT/'GOAL_PORTS.json').read_text())['ports'];self.option_rank=min(range(len(self.terminal_options)),key=lambda k:np.linalg.norm(np.array(self.terminal_options[k]['goal_q'])-self.qg));self.option_tried={self.option_rank};self.option_events=[];self.option_best=np.inf;self.option_last=None;self.option_needs_brake=False
 def feature(self,q):
  self.ctx.guard.positions(q);ee=self.ctx.guard.data.site_xpos[self.ctx.model.site('attachment_site').id].copy()
  return np.r_[ee,self.P@(q-self.memory_origin)]
 def physical_merit(self,state):
  # Fixed denominator over the same seven scalar samples. Duplicate proxy
  # rows and robot sphere density cannot increase an interval's weight.
  return float(sum(max([max(0.,-r['slack']) for r in state['rows'] if r['fraction']==a],default=0.)**2 for a in self.sample_fractions)/len(self.sample_fractions))
 def select_terminal(self):
  oldq=self.qg.copy();oldpoints=self.goal_points.copy();oldcache=self.goal_plane_cache;records=[];feasible=[]
  for rank,port in enumerate(self.terminal_options):
   if rank in self.option_tried:continue
   qg=np.asarray(port['goal_q']);shift=np.r_[np.zeros(3),self.P@(oldq-qg)]
   # Preserve the physical meaning of existing no-return memories. A new
   # terminal pose violating an inherited memory is inadmissible.
   endpoint=np.r_[self.ctx.target,self.P@(qg-self.memory_origin)]
   if any(np.asarray(m['normal'])@endpoint-m['offset']<-1e-10 for m in self.memories):
    records.append(dict(rank=rank,rejected='terminal_incompatible_with_no_return_memory'));self.option_tried.add(rank);continue
   self.qg=qg;self.goal_points=self.ctx.guard.positions(qg).copy();self.goal_plane_cache={};state=Geometry.continuation_witnesses(self,self.q);merit=self.physical_merit(state)
   records.append(dict(rank=rank,physical_merit=merit,physical_deficit=state['deficit'],witnesses=len(state['rows'])))
   feasible.append((merit,rank,port))
  self.qg=oldq;self.goal_points=oldpoints;self.goal_plane_cache=oldcache
  if not feasible:return False
  merit,rank,port=min(feasible,key=lambda x:(x[0],x[1]));newq=np.asarray(port['goal_q']);shift=np.r_[np.zeros(3),self.P@(oldq-newq)]
  self.qg=newq;self.goal_points=self.ctx.guard.positions(newq).copy();self.goal_plane_cache={};self.outward=np.asarray(port['outward']);self.radius=port['linear_radius_m'];self.port=dict(port,posture_matrix=self.P.tolist(),source_rank=rank);self.option_tried.add(rank);before=self.option_rank;self.option_rank=rank
  self.actual_checkpoint=None;self.best_deficit=np.inf;self.option_best=np.inf;self.option_last=self.k;self.witness_anchor=None;self.sample_fractions=np.linspace(0,1,9)[1:-1];self.continuation_rows=np.empty((0,6));self.continuation_bounds=np.empty(0);self.solve_cache={};self.command_epoch+=1
  event=dict(cycle=self.k,previous_rank=before,selected_rank=rank,assessments=records,qg=newq.tolist(),physical_memory_offsets_transformed=0,physical_memories_preserved=len(self.memories));self.option_events.append(event);self.emit('terminal_pose_reselected_after_failed_continuation',**{k:v for k,v in event.items() if k!='cycle'});return True
 def __call__(self,c,s,arrays):
  # Only switch after the original 100-cycle no-progress window. Commands
  # are flushed and braking finishes BEFORE an endpoint changes.
  if self.escape_active and self.mode=='CERT_PREP' and not self.brake_pending:
   raw=Geometry.continuation_witnesses(self,self.q);D=float(np.sqrt(self.physical_merit(raw)))
   if self.option_last is None or D<self.option_best-.0002:self.option_best=D;self.option_last=self.k
   if self.k-self.option_last>=100 and len(self.option_tried)<len(self.terminal_options):
    if np.max(np.abs(self.previous))>1e-8:
     self.option_needs_brake=True;self.brake_pending=True;self.command_epoch+=1;self.expected=None;return np.zeros(6)
    self.select_terminal();self.option_needs_brake=False
  return super().__call__(c,s,arrays)
 def summary(self):
  out=super().summary();out['goal_port']['memory_feature_origin_q']=self.memory_origin.tolist();out.update(t0_revision='consistent_merit_terminal_reselection_two_feature_posture',terminal_progress_metric='RMS of per-fraction maximum physical penetration; same merit for selection and persistence',terminal_options=len(self.terminal_options),terminal_option_switches=self.option_events,terminal_options_tried=sorted(self.option_tried),terminal_pose_values_optimized=False,intermediate_waypoints=0,posture_features_preserved_across_switch=True);return out
