"""Independent enclosure tests: arbitrary translated/rotated ellipsoids and history."""
import numpy as np
from raw_support_intervals import RawSupportIntervals,RawLoewnerJoin
from incremental_proxy_manager import IncrementalMatchedProxyManager

def test_raw_support_complete_ellipsoids_and_history(accumulator=RawSupportIntervals):
    rng=np.random.default_rng(731);acc=accumulator();rep=np.array([[.101,.101,.101]]);saved=[];previous=None
    dirs=rng.normal(size=(1500,3));dirs/=np.linalg.norm(dirs,axis=1)[:,None]
    for frame in range(5):
        points=rep+rng.uniform(-.002,.002,(20,3));mat=rng.normal(size=(20,3,3))*.002;U=mat@mat.transpose(0,2,1)+np.eye(3)*1e-8
        outer,counts=acc.update(points,U,[(0,0,0)],np.zeros(20,dtype=int),rep,np.arange(20),np.array([0]))
        saved.extend(zip(points,U))
        if previous is not None:assert np.linalg.eigvalsh(outer[0]-previous).min()>-1e-15
        previous=outer[0]
        inv=np.linalg.inv(outer[0])
        for p,u in saved:
            val,vec=np.linalg.eigh(u);x=p-rep[0]+(dirs*np.sqrt(val))@vec.T
            assert np.max(np.einsum('ni,ij,nj->n',x,inv,x))<=1+1e-10
        c,R,h,_=acc.cells[(0,0,0)]
        corners=np.array(np.meshgrid(*[[-1,1]]*3)).reshape(3,-1).T*h@R.T
        assert np.max(np.einsum('ni,ij,nj->n',corners,inv,corners))<=1+1e-10
    assert counts[0]==100 and acc.total_samples==100

def test_manager_repeated_observations_preserve_coverage(mode="support_interval_uncertainty"):
    rng=np.random.default_rng(732)
    manager=IncrementalMatchedProxyManager(filter_size=.0075,uncertainty_fusion_mode=mode,minimum_core_semi_axis=.003,certificate_radius_limit=.07,radius_limit_representation='ellipsoid',ellipsoid_cover_mode='adaptive_irredundant')
    for frame in range(3):
        p=np.c_[rng.uniform(.2,.23,60),rng.uniform(.2,.23,60),rng.uniform(.2,.201,60)]
        U=np.tile(np.diag([.001**2,.002**2,.003**2]),(60,1,1));manager.update(p,np.full(60,.003),U)
        assert manager.coverage_audit().all_raw_sample_balls_certified
    assert manager.raw_support_intervals.total_samples==180

if __name__=='__main__':
    test_raw_support_complete_ellipsoids_and_history()
    test_raw_support_complete_ellipsoids_and_history(RawLoewnerJoin)
    test_manager_repeated_observations_preserve_coverage()
    test_manager_repeated_observations_preserve_coverage('raw_join_uncertainty')
    print('Four independent enclosure/history tests passed')
