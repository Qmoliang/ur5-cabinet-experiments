import unittest
import numpy as np
from raw_directional_uncertainty import RawDirectionalJoin
from incremental_proxy_manager import IncrementalMatchedProxyManager

class RawDirectionalTests(unittest.TestCase):
    def test_crossed_views_in_same_frame_remain_distinct(self):
        p=np.array([[.105,.205,.405]]*2);U=np.array([np.diag([36e-6,1e-6,1e-6]),np.diag([1e-6,36e-6,1e-6])])
        manager=IncrementalMatchedProxyManager(filter_size=.0075,uncertainty_fusion_mode='raw_directional_uncertainty',minimum_core_semi_axis=.003,maximum_uncertainty_union_inflation=1.05,certificate_radius_limit=.07,radius_limit_representation='ellipsoid',ellipsoid_cover_mode='adaptive_irredundant')
        manager.update(p,np.zeros(2),U)
        self.assertEqual(len(manager._spatial_cells),1)
        self.assertEqual(len(next(iter(manager._spatial_cells.values())).samples),2)
        self.assertTrue(manager.coverage_audit().all_raw_sample_balls_certified)
        before=manager.snapshot.filtered_uncertainty_shapes.copy()
        manager.update(p,np.zeros(2),U)
        np.testing.assert_array_equal(before,manager.snapshot.filtered_uncertainty_shapes)
        self.assertEqual(manager.coverage_audit().raw_sample_bins,4)
        self.assertTrue(manager.coverage_audit().all_raw_sample_balls_certified)
    def test_complete_translated_sets_and_history(self):
        rng=np.random.default_rng(740);acc=RawDirectionalJoin();rep=np.array([[.2,.3,.4]]);stored=[]
        unit=rng.normal(size=(1000,3));unit/=np.linalg.norm(unit,axis=1)[:,None]
        for k in range(5):
            p=rep+rng.uniform(-.003,.003,(40,3));a=rng.normal(size=(40,3,3))*.002;U=a@a.transpose(0,2,1)+np.eye(3)*1e-8
            codes=acc.bins(U);acc.update(p,U,[(0,0,0)],np.zeros(40,dtype=int),rep);stored.extend(zip(p,U,codes))
            for point,u,code in stored:
                leaf=acc.cells[(0,0,0)][int(code)];ev,R=np.linalg.eigh(u);x=point-rep[0]+(unit*np.sqrt(ev))@R.T;inv=np.linalg.inv(leaf['shape'])
                self.assertLessEqual(float(np.max(np.einsum('ni,ij,nj->n',x,inv,x))),1+1e-10)
        self.assertEqual(sum(l['count'] for l in acc.cells[(0,0,0)].values()),200)
        self.assertLessEqual(len(acc.cells[(0,0,0)]),49)
    def test_bin_sign_and_scale_invariant(self):
        rng=np.random.default_rng(741);a=rng.normal(size=(100,3,3));U=a@a.transpose(0,2,1)+np.eye(3)*1e-3
        np.testing.assert_array_equal(RawDirectionalJoin.bins(U),RawDirectionalJoin.bins(U*4))
        self.assertTrue(np.all(RawDirectionalJoin.bins(np.tile(np.eye(3),(3,1,1)))==48))
if __name__=='__main__':unittest.main()
