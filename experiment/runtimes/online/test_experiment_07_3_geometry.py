"""Independent checks of analytic box planes and hinge displacement bounds."""
import numpy as np,mujoco
from experiment_07_3_geometry import ExactBoxController,distances
from audit_experiment_07_3_continuous import lever_bounds
from official_drawer_07_3 import scene
from model import build_model,build_robot_certificate,set_configuration,certificate_world_positions

def main():
    s=scene();m=build_model(s);d=mujoco.MjData(m);r=build_robot_certificate(m);L=lever_bounds(m,r);rng=np.random.default_rng(73);q=np.array(s.q0);set_configuration(m,d,q);c=ExactBoxController(m,d,s,r)
    for k in range(100):
        q0=q+rng.normal(0,.3,6);q1=q0+rng.normal(0,.1,6)
        set_configuration(m,d,q0);p0=certificate_world_positions(d,r);set_configuration(m,d,q1);p1=certificate_world_positions(d,r)
        assert np.all(np.linalg.norm(p1-p0,axis=1)<=L@np.abs(q1-q0)+1e-12)
    center=np.array([.32,.35,.55]);radius=.04;planes,*_=c.active_planes_for_robot_sphere(0,center,radius)
    for p in planes:
        b=s.boxes[p.obstacle_index];n=p.normal_to_obstacle
        exact_min=n@np.array(b.center)-np.abs(n)@np.array(b.half_size)
        assert abs(exact_min-p.offset)<1e-12
        gap=distances(center[None,:],[b],np.array([radius]))[0,0]
        assert abs(gap-c.safety_margin-p.clearance)<1e-12
    print('100 hinge-displacement checks and analytic supporting-plane checks passed')
if __name__=='__main__':main()
