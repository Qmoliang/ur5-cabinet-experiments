// Strong offline reference: global AABB rejection + SoA eight-lane AVX2 scan.
// Same inclusive float32 ball-AABB predicate and ascending candidate IDs.
#include <immintrin.h>
#include <algorithm>
#include <cstdint>
extern "C" int32_t linear_avx2(const float* lo,const float* hi,int32_t n,
 const float* global_lo,const float* global_hi,const float* centers,const float* radii,
 int32_t nq,int32_t* offsets,int32_t* output,int32_t capacity){
 int32_t written=0;offsets[0]=0;
 for(int32_t q=0;q<nq;q++){
  const float* c=centers+3*q;const float r=radii[q],r2=r*r;
  bool outside=false;for(int d=0;d<3;d++)outside|=(c[d]-r>global_hi[d]||c[d]+r<global_lo[d]);
  if(outside){offsets[q+1]=written;continue;}
  const __m256 cx=_mm256_set1_ps(c[0]),cy=_mm256_set1_ps(c[1]),cz=_mm256_set1_ps(c[2]),zero=_mm256_setzero_ps(),rr=_mm256_set1_ps(r2);
  int32_t i=0;
  for(;i+8<=n;i+=8){
   auto dx=_mm256_max_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_loadu_ps(lo+i),cx),_mm256_sub_ps(cx,_mm256_loadu_ps(hi+i))),zero);
   auto dy=_mm256_max_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_loadu_ps(lo+n+i),cy),_mm256_sub_ps(cy,_mm256_loadu_ps(hi+n+i))),zero);
   auto dz=_mm256_max_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_loadu_ps(lo+2*n+i),cz),_mm256_sub_ps(cz,_mm256_loadu_ps(hi+2*n+i))),zero);
   auto d=_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx,dx),_mm256_mul_ps(dy,dy)),_mm256_mul_ps(dz,dz));
   const int bits=_mm256_movemask_ps(_mm256_cmp_ps(d,rr,_CMP_LE_OQ));
   for(int j=0;j<8;j++)if(bits&(1<<j)){if(written>=capacity)return -2;output[written++]=i+j;}
  }
  for(;i<n;i++){float d2=0;for(int k=0;k<3;k++){float d=std::max(std::max(lo[k*n+i]-c[k],c[k]-hi[k*n+i]),0.0f);d2=d2+d*d;}if(d2<=r2){if(written>=capacity)return -2;output[written++]=i;}}
  offsets[q+1]=written;
 }
 return written;
}
