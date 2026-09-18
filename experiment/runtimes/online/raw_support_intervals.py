"""Causal support-box accumulator for complete translated measurement ellipsoids.

For each stable CenterVox representative c and first-frame orthonormal R,
abs(R.T @ (p-c)) + sqrt(diag(R.T U R)) is an exact symmetric coordinate
bound for p+E(U). Componentwise maxima retain all past observations.
The box is contained in E(R diag(3 h^2) R.T), since sum x_i^2/(3h_i^2)<=1.
No measurement intersection or variance reduction is used.
"""
import numpy as np

class RawSupportIntervals:
    def __init__(self):
        self.cells={};self.total_samples=0;self.max_box_corner_value=0.;self.minimum_interval_slack=float('inf')
    def update(self,points,shapes,keys,inverse,representatives,order,first):
        n=len(keys);rot=np.empty((n,3,3));half=np.zeros((n,3));counts=np.zeros(n,dtype=np.int64)
        ends=np.r_[first[1:],len(order)]
        for i,key in enumerate(keys):
            old=self.cells.get(key)
            if old is None:
                ix=order[first[i]:ends[i]]
                # Fixed frame chosen once from measurement geometry, no target/true map.
                delta=points[ix]-representatives[i]
                moment=np.mean(shapes[ix],axis=0)+delta.T@delta/len(ix)
                _,rot[i]=np.linalg.eigh(moment)
            else:
                assert np.array_equal(old[0],representatives[i]),'support frame requires stable representative'
                rot[i]=old[1];half[i]=old[2];counts[i]=old[3]
        rawrot=rot[inverse]
        local=np.einsum('ni,nik->nk',points-representatives[inverse],rawrot)
        support=np.sqrt(np.maximum(np.einsum('nik,nij,njk->nk',rawrot,shapes,rawrot),0))
        required=np.abs(local)+support
        current=np.maximum.reduceat(required[order],first,axis=0)
        half=np.maximum(half,current);half=np.maximum(half,1e-10)
        slack=float(np.min(half[inverse]-required))
        if slack < -1e-12:raise AssertionError('raw measurement escaped support interval')
        self.minimum_interval_slack=min(self.minimum_interval_slack,slack)
        counts+=np.bincount(inverse,minlength=n)
        outer=np.einsum('nik,nk,njk->nij',rot,3*half**2,rot)
        self.max_box_corner_value=max(self.max_box_corner_value,float(np.max(np.sum(half**2/(3*half**2),axis=1))))
        for i,key in enumerate(keys):self.cells[key]=(representatives[i].copy(),rot[i].copy(),half[i].copy(),int(counts[i]))
        self.total_samples+=len(points)
        return outer,counts

class RawLoewnerJoin:
    """One persistent frame and one diagonal-dominance union of raw translated U."""
    def __init__(self):
        self.cells={};self.total_samples=0;self.minimum_psd_slack=float('inf')
        self.minimum_interval_slack=0.;self.max_box_corner_value=None
    def update(self,points,shapes,keys,inverse,representatives,order,first):
        from pointcloud_proxy import minkowski_outer_shapes
        delta=points-representatives[inverse]
        translated=minkowski_outer_shapes(shapes,np.einsum('ni,nj->nij',delta,delta))
        n=len(keys);rot=np.empty((n,3,3));bounds=np.zeros((n,3));counts=np.zeros(n,dtype=np.int64);ends=np.r_[first[1:],len(order)]
        for i,key in enumerate(keys):
            old=self.cells.get(key)
            if old is None:
                _,rot[i]=np.linalg.eigh(np.mean(translated[order[first[i]:ends[i]]],axis=0))
            else:
                assert np.array_equal(old[0],representatives[i])
                rot[i]=old[1];bounds[i]=3*old[2]**2;counts[i]=old[3]
        R=rot[inverse];B=np.einsum('nji,njk,nkl->nil',R,translated,R)
        diagonal=np.diagonal(B,axis1=1,axis2=2)
        required=diagonal+np.sum(np.abs(B),axis=2)-np.abs(diagonal)
        bounds=np.maximum(bounds,np.maximum.reduceat(required[order],first,axis=0))
        bounds+=np.maximum(bounds.max(axis=1),1e-18)[:,None]*2e-14+1e-18
        outer=np.einsum('nik,nk,njk->nij',rot,bounds,rot)
        # Full matrix check, not sampled directions, for every raw translated ellipsoid.
        slack=float(np.linalg.eigvalsh(outer[inverse]-translated)[:,0].min())
        if slack < -1e-12:raise AssertionError('raw translated U escaped the persistent Loewner join')
        self.minimum_psd_slack=min(self.minimum_psd_slack,slack)
        counts+=np.bincount(inverse,minlength=n)
        for i,key in enumerate(keys):self.cells[key]=(representatives[i].copy(),rot[i].copy(),np.sqrt(bounds[i]/3),int(counts[i]))
        self.total_samples+=len(points)
        return outer,counts
