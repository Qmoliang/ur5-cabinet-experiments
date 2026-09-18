"""Numerical checks using the actual frozen v4.3 / v4.4c proxy archives."""
import json
import numpy as np
import mujoco
from neo_online import HERE,BASE,OnlineNEO,quadprog
from freeze_v4_4_v4_3_formal_baselines import SPHERE_ROOT,ELLIPSOID_ROOT,one_run
from protocol_drawer_scene import formal_drawer_camera_quarter_scene
from model import build_robot_certificate,set_configuration,certificate_world_state
from run_study import dump

def main():
    checks=[]
    for representation,root in [('sphere',SPHERE_ROOT),('ellipsoid',ELLIPSOID_ROOT)]:
        folder,_=one_run(root); archive=np.load(folder/'final_causal_proxies.npz')
        scene=formal_drawer_camera_quarter_scene();model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
        data=mujoco.MjData(model);set_configuration(model,data,np.array(scene.q0));robot=build_robot_certificate(model)
        ctrl=OnlineNEO(model,data,scene,robot,archive['centers'],representation=representation,
                       obstacle_radii=archive['sphere_radii'] if representation=='sphere' else None,
                       obstacle_shapes=archive['ellipsoid_shapes'] if representation=='ellipsoid' else None,
                       obstacle_uncertainty_shapes=archive['proxy_uncertainty_shapes'] if representation=='ellipsoid' else None,
                       obstacle_offsets=archive['uncertainty_offsets'])
        q=data.qpos[:6].copy();pos,jac,r=certificate_world_state(model,data,robot)
        # Full versus MVT for two physical wrist spheres, including directional U.
        chosen=np.array([50,58]);ri,oi,d,n,it,res=ctrl.geometry(pos[chosen],r[chosen],full=True)
        ri2,oi2,d2,n2,_,_=ctrl.geometry(pos[chosen],r[chosen])
        expected=set(zip(ri[d<.30],oi[d<.30]));got=set(zip(ri2[d2<.30],oi2[d2<.30]))
        assert expected.issubset(got)
        direction=np.array([.2,-.3,.1,.15,-.1,.25]);step=1e-6;dist=[]
        for sign in [1,-1]:
            set_configuration(model,data,q+sign*step*direction);p,_,_=certificate_world_state(model,data,robot)
            dist.append(ctrl.geometry(p[chosen],r[chosen],full=True)[2])
        exact=-np.einsum('pi,pij,j->p',n,jac[chosen[ri]],direction)
        deriv_error=float(np.max(np.abs((dist[0]-dist[1])/(2*step)-exact)))
        assert deriv_error<2e-5,(representation,deriv_error)
        set_configuration(model,data,q);H,g,A,lo,hi,info=ctrl.assemble(np.array(scene.waypoints[-1]))
        nonzero=np.linalg.norm(A,axis=1)>1e-14; aa=A[nonzero];ll=lo[nonzero];hh=hi[nonzero]
        upper=np.isfinite(hh);lower=np.isfinite(ll);C=np.vstack([-aa[upper],aa[lower]]).T;b=np.r_[-hh[upper],ll[lower]]
        u=quadprog.solve_qp(H,-g,C,b)[0]
        H9=np.diag(np.r_[np.full(6,.01*info['e']),np.ones(3)])
        g9=np.r_[-info['e']*info['jm'],np.zeros(3)]
        equality=np.c_[info['J'],np.eye(3)]
        C9=np.c_[equality.T,np.r_[C,np.zeros((3,C.shape[1]))]]
        x=quadprog.solve_qp(H9,-g9,C9,np.r_[info['v'],b],meq=3)[0]
        qp_error=float(np.max(np.abs(u-x[:6])));assert qp_error<1e-7
        checks.append(dict(representation=representation,proxy_count=len(archive['centers']),
                           full_pair_geometry_oracle_passed=True,distance_derivative_error=deriv_error,
                           support_residual=res,explicit_slack_equivalence_max_rad_s=qp_error,
                           index_audits=ctrl.index_audits))
        ctrl.close()
    dump(HERE/'preflight.json',dict(passed=True,checks=checks));print(json.dumps(checks),flush=True)

if __name__=='__main__':main()
