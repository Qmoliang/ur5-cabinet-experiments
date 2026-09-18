"""Causal raw-direction leaves; complete error sets, one spatial map.

Bins select which error matrices share an outer certificate. They never
approximate directions in a distance calculation. Each leaf analytically
contains every translated input matrix assigned to it (PSD inequality).
"""
import numpy as np
from pointcloud_proxy import minkowski_outer_shapes

class RawDirectionalJoin:
    MAX_LEAVES_PER_VOXEL=49
    def __init__(self):
        self.cells={};self.total_samples=0;self.minimum_psd_slack=float('inf');self.generation=-1
    @staticmethod
    def bins(shapes):
        val,vec=np.linalg.eigh(shapes);axis=vec[:,:,-1];face=np.argmax(np.abs(axis),axis=1);dominant=axis[np.arange(len(axis)),face]
        canonical=axis/dominant[:,None]
        rest=np.array([[1,2],[0,2],[0,1]])[face]
        uv=np.take_along_axis(canonical,rest,axis=1)
        ij=np.clip(np.floor((uv+1)*2).astype(int),0,3)
        ids=face*16+ij[:,0]*4+ij[:,1]
        ids[val[:,-1] <= 2*np.maximum(val[:,0],1e-18)]=48
        return ids
    def update(self,points,shapes,keys,inverse,representatives):
        self.generation+=1
        codes=self.bins(shapes);combined=np.c_[inverse,codes];unique,owners=np.unique(combined,axis=0,return_inverse=True)
        order=np.argsort(owners,kind='stable');first=np.r_[0,np.flatnonzero(np.diff(owners[order]))+1];ends=np.r_[first[1:],len(order)]
        delta=points-representatives[inverse];translated=minkowski_outer_shapes(shapes,np.einsum('ni,nj->nij',delta,delta))
        n=len(unique);R=np.empty((n,3,3));D=np.zeros((n,3));counts=np.zeros(n,dtype=np.int64);generations=[]
        for i,(spatial,code) in enumerate(unique):
            key=keys[spatial];old=self.cells.get(key,{}).get(int(code))
            if old is None:
                _,R[i]=np.linalg.eigh(np.mean(translated[order[first[i]:ends[i]]],axis=0));generations.append(frozenset({self.generation}))
            else:
                assert np.array_equal(old['center'],representatives[spatial])
                R[i]=old['R'];D[i]=old['D'];counts[i]=old['count'];generations.append(old['generations']|{self.generation})
        frames=R[owners];B=np.einsum('nji,njk,nkl->nil',frames,translated,frames);diag=np.diagonal(B,axis1=1,axis2=2)
        bound=diag+np.sum(np.abs(B),axis=2)-np.abs(diag)
        current=np.maximum.reduceat(bound[order],first,axis=0)
        current+=np.maximum(current.max(axis=1),1e-18)[:,None]*2e-14+1e-18
        D=np.maximum(D,current)
        outer=np.einsum('nik,nk,njk->nij',R,D,R)
        slack=float(np.linalg.eigvalsh(outer[owners]-translated)[:,0].min())
        if slack < -1e-12:raise AssertionError('raw directional leaf failed full PSD containment')
        self.minimum_psd_slack=min(self.minimum_psd_slack,slack);self.total_samples+=len(points);counts+=np.bincount(owners,minlength=n)
        for i,(spatial,code) in enumerate(unique):
            key=keys[spatial];self.cells.setdefault(key,{})[int(code)]=dict(center=representatives[spatial].copy(),R=R[i].copy(),D=D[i].copy(),shape=outer[i].copy(),count=int(counts[i]),generations=generations[i])
        return [self.cells[key] for key in keys]
