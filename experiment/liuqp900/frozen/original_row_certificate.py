"""Assemble exactly the existing original LiuQP rows without solving its QP."""
import inspect,hashlib,textwrap
import numpy as np
import protocol_liuqp_controller as original
from pilot_adapter import build,online
from scipy.optimize import linprog

def build_assembler(task,representation='ellipsoid'):
 ctx=build(task,representation);c=ctx.controller
 source=next(textwrap.dedent(inspect.getsource(cls.solve)) for cls in type(c).__mro__ if hasattr(cls,'solve') and 'setup_started = time.perf_counter()' in inspect.getsource(cls.solve))
 # This is a literal prefix of the existing solve method. Constraint formulas,
 # pruning, margins, bounds and row identities are neither reimplemented nor changed.
 marker='    setup_started = time.perf_counter()'
 prefix=source[:source.index(marker)];prefix=prefix.replace('def solve(', 'def assemble_original_rows(',1)
 prefix+='    return self.last_qp_arrays, qp_full_row_identity_tuple, ee_jacobian.copy()\n'
 namespace=dict(vars(original));exec(compile(prefix,'<unchanged_original_row_prefix>','exec'),namespace)
 base_function=namespace['assemble_original_rows']
 mixed_source=textwrap.dedent(inspect.getsource(type(c).solve));mixed_prefix=mixed_source[:mixed_source.index('    try:v,m=super().solve(target)')].replace('def solve(', 'def assemble_mixed(',1)
 mixed_prefix+='    try:return base_function(self,target)\n    finally:self.extra_rows=previous;self.extra_near_rows=previous_near\n'
 mixed_namespace=dict(vars(inspect.getmodule(type(c))));mixed_namespace['base_function']=base_function;exec(compile(mixed_prefix,'<unchanged_frontier_row_prefix>','exec'),mixed_namespace);function=mixed_namespace['assemble_mixed']
 class Assembler:
  def __init__(self):self.ctx=ctx;self.calls=0;self.source_hash=hashlib.sha256(source.encode()).hexdigest()
  def rows(self,q,previous,target):
   online.set_configuration(ctx.model,ctx.data,q);c.applied_velocity=np.asarray(previous).copy();c.previous_velocity=np.asarray(previous).copy();self.calls+=1
   return function(c,target)
 return Assembler()

def check_rows(arrays,v,tol=1e-7):
 _,_,A,l,u=arrays;value=A@v;return float(max(0.,np.max(l-value),np.max(value-u)))

def feasible(arrays):
 _,_,A,l,u=arrays;up=np.isfinite(u);lo=np.isfinite(l);z=linprog(np.zeros(6),A_ub=np.vstack([A[up],-A[lo]]),b_ub=np.r_[u[up],-l[lo]],bounds=[(None,None)]*6,method='highs',options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9));return dict(feasible=bool(z.success),status=int(z.status),violation=None if not z.success else check_rows(arrays,z.x))
