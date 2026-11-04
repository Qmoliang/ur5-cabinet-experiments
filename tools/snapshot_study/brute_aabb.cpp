// Independent scalar C++ reference for the frozen float32 ball-AABB predicate.
// Benchmark-only code. It is never loaded by a running experiment/controller.
#include <algorithm>
#include <cstdint>
extern "C" int32_t brute_batch(const float* lo,const float* hi,int32_t n,
 const float* centers,const float* radii,int32_t nq,int32_t* offsets,
 int32_t* output,int32_t capacity) {
 int32_t written=0; offsets[0]=0;
 for(int32_t q=0;q<nq;q++) {
  const float* c=centers+3*q;const float r2=radii[q]*radii[q];
  for(int32_t i=0;i<n;i++) {
   const float x=std::max(std::max(lo[3*i]-c[0],c[0]-hi[3*i]),0.0f);
   const float y=std::max(std::max(lo[3*i+1]-c[1],c[1]-hi[3*i+1]),0.0f);
   const float z=std::max(std::max(lo[3*i+2]-c[2],c[2]-hi[3*i+2]),0.0f);
   float d=x*x;d=d+y*y;d=d+z*z;
   if(d<=r2){if(written>=capacity)return -2;output[written++]=i;}
  }
  offsets[q+1]=written;
 }
 return written;
}
