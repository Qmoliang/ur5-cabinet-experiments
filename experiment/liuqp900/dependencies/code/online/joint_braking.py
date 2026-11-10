"""Joint travel viability for zero-order-held velocity commands and bounded acceleration."""
from __future__ import annotations
import numpy as np

class JointLimitInfeasible(RuntimeError):
    """No velocity can satisfy the unchanged joint and acceleration limits."""

# Inward numerical reserve, in radians; it does not enlarge the joint range.
BRAKING_RESERVE = 1e-10
VELOCITY_RESERVE = 1e-9  # inward bias prevents cancellation at a braking boundary

def stopping_distance(speed, dt, acceleration):
    """Travel from executing speed now, then braking by acceleration*dt each step."""
    speed=np.abs(np.asarray(speed,dtype=float));step=float(acceleration)*float(dt)
    if not (dt>0 and acceleration>0):raise ValueError('positive dt and acceleration required')
    n=np.maximum(1.,np.ceil(speed/step))
    return float(dt)*(n*speed-step*n*(n-1.)/2.)

def braking_speed_limit(distance, dt, acceleration):
    """Exact inverse of the piecewise-linear discrete stopping distance."""
    if not (dt>0 and acceleration>0):raise ValueError('positive dt and acceleration required')
    distance=np.maximum(np.asarray(distance,dtype=float),0.);step=float(acceleration)*float(dt)
    n=np.maximum(1.,np.ceil((np.sqrt(1.+8.*distance/(float(dt)*step))-1.)/2.))
    return distance/(float(dt)*n)+step*(n-1.)/2.

def joint_velocity_bounds(q, previous, joint_min, joint_max, velocity_limits, dt, acceleration=None, padding=.015):
    q=np.asarray(q,dtype=float);previous=np.asarray(previous,dtype=float);vmax=np.asarray(velocity_limits,dtype=float)
    low=np.asarray(joint_min,dtype=float)+padding;high=np.asarray(joint_max,dtype=float)-padding
    if not dt>0 or not np.all(np.isfinite(q)) or not np.all(np.isfinite(previous)):raise ValueError('invalid joint state or timestep')
    if np.any(q<low-1e-12) or np.any(q>high+1e-12):raise JointLimitInfeasible('current joint position outside padded range')
    if acceleration is None:
        return np.maximum(-vmax,(low-q)/dt),np.minimum(vmax,(high-q)/dt)
    if not np.isfinite(acceleration) or acceleration<=0:raise ValueError('positive finite acceleration required')
    lower=np.maximum(-vmax,-np.maximum(0.,braking_speed_limit(q-low-BRAKING_RESERVE,dt,acceleration)-VELOCITY_RESERVE))
    upper=np.minimum(vmax,np.maximum(0.,braking_speed_limit(high-q-BRAKING_RESERVE,dt,acceleration)-VELOCITY_RESERVE))
    step=acceleration*dt
    lower=np.maximum(lower,previous-step);upper=np.minimum(upper,previous+step)
    if np.any(lower>upper):
        joints=np.flatnonzero(lower>upper).tolist()
        raise JointLimitInfeasible(f'joint braking/acceleration intersection empty at joints {joints}')
    return lower,upper
