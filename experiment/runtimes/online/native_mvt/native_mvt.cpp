#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <malloc.h>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <immintrin.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define MVT_API extern "C" __declspec(dllexport)
#else
#define MVT_API extern "C"
#endif

// The formal controller owns one native kernel instance per process.  Keep
// its OpenMP width explicit and auditable so CPU isolation can match physical
// cores instead of oversubscribing an affinity partition.
static int32_t g_ellipsoid_pair_threads = 8;
static uint64_t g_ellipsoid_pair_affinity_mask = 0;

MVT_API int32_t ellipsoid_set_pair_threads(int32_t value) {
  if (value < 1 || value > 64) return -1;
  g_ellipsoid_pair_threads = value;
  return value;
}

MVT_API int32_t ellipsoid_get_pair_threads() {
  return g_ellipsoid_pair_threads;
}

MVT_API uint64_t ellipsoid_set_pair_affinity_mask(uint64_t value) {
  g_ellipsoid_pair_affinity_mask = value;
  return value;
}

MVT_API uint64_t ellipsoid_get_pair_affinity_mask() {
  return g_ellipsoid_pair_affinity_mask;
}

template <class T, std::size_t Alignment>
class AlignedAllocator {
 public:
  using value_type = T;
  AlignedAllocator() noexcept = default;
  template <class U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
  T* allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T)) throw std::bad_alloc();
#ifdef _WIN32
    void* p = _aligned_malloc(n * sizeof(T), Alignment);
    if (!p) throw std::bad_alloc();
    return static_cast<T*>(p);
#else
    void* p = nullptr;
    if (posix_memalign(&p, Alignment, n * sizeof(T))) throw std::bad_alloc();
    return static_cast<T*>(p);
#endif
  }
  void deallocate(T* p, std::size_t) noexcept {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
  }
  template <class U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
};
template <class T1, std::size_t A1, class T2, std::size_t A2>
bool operator==(const AlignedAllocator<T1, A1>&, const AlignedAllocator<T2, A2>&) { return A1 == A2; }
template <class T1, std::size_t A1, class T2, std::size_t A2>
bool operator!=(const AlignedAllocator<T1, A1>& a, const AlignedAllocator<T2, A2>& b) { return !(a == b); }

using AlignedFloats = std::vector<float, AlignedAllocator<float, 32>>;

struct VoxelBuild {
  std::vector<int32_t> indices;
  float lo[3] = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
  float hi[3] = {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
};

struct NativeMVT {
  int32_t n = 0;
  float voxel = 0.05f;
  float origin[3]{};
  int32_t dims[3]{};
  float global_lo[3]{};
  float global_hi[3]{};

  std::vector<int32_t> x_table;
  std::vector<int32_t> y_pool;
  std::vector<int32_t> z_pool;
  std::vector<int32_t> voxel_start;
  std::vector<int32_t> voxel_count;
  std::vector<int32_t> index_pool;
  AlignedFloats voxel_lo[3], voxel_hi[3];
  AlignedFloats proxy_lo[3], proxy_hi[3];
  mutable std::vector<uint32_t> marks;
  mutable uint32_t epoch = 1;

  int clamp_index(float value, int axis) const {
    int idx = static_cast<int>(std::floor((value - origin[axis]) / voxel));
    return std::max(0, std::min(dims[axis] - 1, idx));
  }
  int64_t flat(int x, int y, int z) const {
    return (static_cast<int64_t>(x) * dims[1] + y) * dims[2] + z;
  }
};

// Genuine VCC-style multilevel voxel table.  Every proxy is stored in exactly
// one cell at exactly one level.  A proxy is assigned to the smallest level
// whose cell width is strictly larger than
//
//     maximum_query_half_extent + maximum_proxy_half_extent.
//
// Therefore, if a query AABB intersects a proxy AABB, their center-cell
// indices can differ by at most one on every axis.  Looking up the center cell
// plus its 26 neighbors at every level is consequently a conservative
// broadphase with no proxy replication and no false negatives.
struct MultilevelVoxelLevel {
  float voxel = 0.0f;
  std::unordered_map<uint64_t, int32_t> cell_ids;
  std::vector<int32_t> cell_start;
  std::vector<int32_t> cell_count;
  std::vector<int32_t> index_pool;
  // AABBs are duplicated once, in the exact order of index_pool.  Each proxy
  // belongs to exactly one level/cell, so this is O(N) storage and lets AVX2
  // use contiguous loads instead of six scattered gathers per eight proxies.
  AlignedFloats packed_lo[3], packed_hi[3];
  AlignedFloats cell_lo[3], cell_hi[3];
};

struct NativeMultilevelMVT {
  int32_t n = 0;
  float base_voxel = 0.0f;
  float maximum_query_half = 0.0f;
  float global_lo[3]{};
  float global_hi[3]{};
  std::vector<MultilevelVoxelLevel> levels;
  AlignedFloats proxy_lo[3], proxy_hi[3];
};

struct NativeOccupancy {
  float voxel = 0.015f;
  std::unordered_map<uint64_t, uint8_t> states;
};

struct NativeIncrementalOccupancy {
  double voxel = 0.015;
  int8_t free_decrement = 1;
  int8_t occupied_increment = 4;
  int8_t minimum_score = -8;
  int8_t maximum_score = 8;
  int8_t occupied_threshold = 2;
  int8_t free_threshold = -1;
  int32_t frame_index = -1;
  std::unordered_map<uint64_t, int8_t> scores;
  bool has_calibrated_free_aabb = false;
  double calibrated_free_lo[3]{};
  double calibrated_free_hi[3]{};
  int64_t sparse_free_voxels = 0;
  int64_t sparse_occupied_voxels = 0;
  // State-change journal for causal visualization/logging.  Scores that move
  // within the same FREE/UNKNOWN/OCCUPIED class are intentionally omitted:
  // they cannot change any three-state query result.
  std::unordered_set<uint64_t> dirty_state_keys;
  bool state_journal_enabled = false;
};

static uint64_t pack_voxel(int32_t x, int32_t y, int32_t z) {
  constexpr int64_t bias = int64_t{1} << 20;
  constexpr uint64_t mask = (uint64_t{1} << 21) - 1;
  const uint64_t ux = static_cast<uint64_t>(static_cast<int64_t>(x) + bias) & mask;
  const uint64_t uy = static_cast<uint64_t>(static_cast<int64_t>(y) + bias) & mask;
  const uint64_t uz = static_cast<uint64_t>(static_cast<int64_t>(z) + bias) & mask;
  return ux | (uy << 21) | (uz << 42);
}

static int8_t incremental_state(const NativeIncrementalOccupancy* map,
                                uint64_t key) {
  const auto found = map->scores.find(key);
  if (found != map->scores.end() &&
      found->second >= map->occupied_threshold) return 2;
  if (found != map->scores.end() && found->second <= map->free_threshold)
    return 1;
  if (map->has_calibrated_free_aabb) {
    constexpr int64_t bias = int64_t{1} << 20;
    constexpr uint64_t mask = (uint64_t{1} << 21) - 1;
    bool inside = true;
    for (int axis = 0; axis < 3; ++axis) {
      const int32_t coordinate = static_cast<int32_t>(
          static_cast<int64_t>((key >> (21 * axis)) & mask) - bias);
      const double cell_lo = static_cast<double>(coordinate) * map->voxel;
      const double cell_hi = static_cast<double>(coordinate + 1) * map->voxel;
      inside = inside &&
          cell_lo >= map->calibrated_free_lo[axis] - 1.0e-12 &&
          cell_hi <= map->calibrated_free_hi[axis] + 1.0e-12;
    }
    if (inside) return 1;
  }
  return 0;
}

static int8_t incremental_score_state(
    const NativeIncrementalOccupancy* map, int score) {
  if (score >= map->occupied_threshold) return 2;
  if (score <= map->free_threshold) return 1;
  return 0;
}

static void incremental_set_score(
    NativeIncrementalOccupancy* map, uint64_t key, int score) {
  const auto found = map->scores.find(key);
  const int8_t previous = found == map->scores.end()
      ? 0 : incremental_score_state(map, found->second);
  score = std::max<int>(map->minimum_score,
                        std::min<int>(map->maximum_score, score));
  map->scores[key] = static_cast<int8_t>(score);
  const int8_t current = incremental_score_state(map, score);
  map->sparse_free_voxels += (current == 1) - (previous == 1);
  map->sparse_occupied_voxels += (current == 2) - (previous == 2);
  if (map->state_journal_enabled && current != previous)
    map->dirty_state_keys.insert(key);
}

static void incremental_add_score(NativeIncrementalOccupancy* map,
                                  uint64_t key, int delta) {
  const auto found = map->scores.find(key);
  int score = (found == map->scores.end() ? 0 : found->second) + delta;
  incremental_set_score(map, key, score);
}

template <class Visitor>
static int64_t visit_conservative_ball(double voxel, const double center[3],
                                       double radius, Visitor&& visitor) {
  const double half_diagonal = 0.86602540378443864676 * voxel;
  const double cover = std::max(0.0, radius) + half_diagonal;
  const double cover2 = static_cast<double>(cover) * cover;
  int32_t base[3];
  for (int axis = 0; axis < 3; ++axis)
    base[axis] = static_cast<int32_t>(std::floor(center[axis] / voxel));
  const int32_t reach =
      static_cast<int32_t>(std::ceil(cover / voxel)) + 1;
  int64_t visited = 0;
  for (int32_t x = base[0] - reach; x <= base[0] + reach; ++x) {
    const double dx = (static_cast<double>(x) + 0.5) * voxel - center[0];
    for (int32_t y = base[1] - reach; y <= base[1] + reach; ++y) {
      const double dy = (static_cast<double>(y) + 0.5) * voxel - center[1];
      for (int32_t z = base[2] - reach; z <= base[2] + reach; ++z) {
        const double dz = (static_cast<double>(z) + 0.5) * voxel - center[2];
        if (dx * dx + dy * dy + dz * dz > cover2) continue;
        visitor(pack_voxel(x, y, z));
        ++visited;
      }
    }
  }
  return visited;
}

static int32_t unpack_voxel_axis(uint64_t packed, int axis) {
  constexpr int64_t bias = int64_t{1} << 20;
  constexpr uint64_t mask = (uint64_t{1} << 21) - 1;
  return static_cast<int32_t>(static_cast<int64_t>((packed >> (21 * axis)) & mask) - bias);
}

static bool intersects(const float alo[3], const float ahi[3], const float blo[3], const float bhi[3]) {
  return alo[0] <= bhi[0] && ahi[0] >= blo[0] &&
         alo[1] <= bhi[1] && ahi[1] >= blo[1] &&
         alo[2] <= bhi[2] && ahi[2] >= blo[2];
}

static bool invert_spd3(const double q[9], double inverse[9]) {
  const double a=q[0], b=q[1], c=q[2], d=q[4], e=q[5], f=q[8];
  const double determinant = a*(d*f-e*e)-b*(b*f-c*e)+c*(b*e-c*d);
  if (!(determinant > 1.0e-30)) return false;
  inverse[0]=(d*f-e*e)/determinant;
  inverse[1]=(c*e-b*f)/determinant;
  inverse[2]=(b*e-c*d)/determinant;
  inverse[3]=inverse[1];
  inverse[4]=(a*f-c*c)/determinant;
  inverse[5]=(b*c-a*e)/determinant;
  inverse[6]=inverse[2];
  inverse[7]=inverse[5];
  inverse[8]=(a*d-b*b)/determinant;
  return true;
}

static bool solve_small(double matrix[3][3], double rhs[3], int n,
                        double solution[3]) {
  for (int column=0; column<n; ++column) {
    int pivot=column;
    for (int row=column+1; row<n; ++row)
      if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column]))
        pivot=row;
    if (std::abs(matrix[pivot][column]) <= 1.0e-18) return false;
    if (pivot != column) {
      for (int k=column; k<n; ++k) std::swap(matrix[pivot][k],matrix[column][k]);
      std::swap(rhs[pivot],rhs[column]);
    }
    for (int row=column+1; row<n; ++row) {
      const double factor=matrix[row][column]/matrix[column][column];
      for (int k=column; k<n; ++k) matrix[row][k]-=factor*matrix[column][k];
      rhs[row]-=factor*rhs[column];
    }
  }
  for (int row=n-1; row>=0; --row) {
    double value=rhs[row];
    for (int k=row+1; k<n; ++k) value-=matrix[row][k]*solution[k];
    solution[row]=value/matrix[row][row];
  }
  return true;
}

// Exact convex test for E(c,Q) intersecting an axis-aligned voxel box.  The
// minimum of (x-c)'Q^-1(x-c) over a 3-D box is obtained by enumerating the
// lower/free/upper active state of each coordinate (3^3 cases).
static bool ellipsoid_intersects_box(const double center[3], const double q[9],
                                     const double box_lo[3],
                                     const double box_hi[3]) {
  double h[9];
  if (!invert_spd3(q,h)) return true;
  double lo[3],hi[3];
  for (int axis=0; axis<3; ++axis) {
    lo[axis]=box_lo[axis]-center[axis];
    hi[axis]=box_hi[axis]-center[axis];
  }
  double best=std::numeric_limits<double>::infinity();
  for (int code=0; code<27; ++code) {
    int state[3]; int value=code;
    int free_axes[3],free_count=0;
    double y[3]{};
    for (int axis=0; axis<3; ++axis) {
      state[axis]=value%3; value/=3;
      if (state[axis]==0) free_axes[free_count++]=axis;
      else y[axis]=(state[axis]==1 ? lo[axis] : hi[axis]);
    }
    if (free_count) {
      double matrix[3][3]{}; double rhs[3]{}; double solution[3]{};
      for (int row=0; row<free_count; ++row) {
        const int i=free_axes[row];
        for (int column=0; column<free_count; ++column)
          matrix[row][column]=h[3*i+free_axes[column]];
        for (int fixed=0; fixed<3; ++fixed)
          if (state[fixed]!=0) rhs[row]-=h[3*i+fixed]*y[fixed];
      }
      if (!solve_small(matrix,rhs,free_count,solution)) continue;
      bool feasible=true;
      for (int row=0; row<free_count; ++row) {
        const int axis=free_axes[row]; y[axis]=solution[row];
        feasible = feasible && y[axis]>=lo[axis]-1.0e-12 &&
                   y[axis]<=hi[axis]+1.0e-12;
      }
      if (!feasible) continue;
    }
    double objective=0.0;
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j) objective+=y[i]*h[3*i+j]*y[j];
    best=std::min(best,objective);
  }
  return best <= 1.0 + 1.0e-10;
}

template <class Visitor>
static int64_t visit_conservative_ellipsoid(
    double voxel, const double center[3], const double shape[9],
    Visitor&& visitor) {
  // If a voxel cube intersects E(Q), its center belongs to
  // E(Q) + B(0,sqrt(3)*voxel/2).  Enclose that Minkowski sum with the same
  // trace-balanced Young ellipsoid used by the online controller, then use
  // one quadratic form per voxel center.  This is conservative and retains
  // directional narrow axes, while avoiding 27 active-set solves per endpoint
  // voxel during map integration.  Exact ellipsoid/box tests remain in the
  // immutable occupancy query used by the certification oracle.
  const double half_diagonal = 0.86602540378443864676 * voxel;
  const double ball_variance = half_diagonal * half_diagonal;
  const double shape_trace = std::max(
      1.0e-18, shape[0] + shape[4] + shape[8]);
  const double beta = std::sqrt(3.0 * ball_variance / shape_trace);
  double outer[9];
  for (int i = 0; i < 9; ++i)
    outer[i] = (1.0 + beta) * shape[i];
  const double ball_scale = (1.0 + 1.0 / beta) * ball_variance;
  outer[0] += ball_scale;
  outer[4] += ball_scale;
  outer[8] += ball_scale;
  double inverse[9];
  if (!invert_spd3(outer, inverse)) return 0;
  int32_t lower[3], upper[3];
  for (int axis = 0; axis < 3; ++axis) {
    const double extent = std::sqrt(
        std::max(0.0, outer[3 * axis + axis]));
    lower[axis] = static_cast<int32_t>(
        std::floor((center[axis] - extent) / voxel));
    upper[axis] = static_cast<int32_t>(
        std::floor((center[axis] + extent) / voxel));
  }
  int64_t visited = 0;
  for (int32_t x = lower[0]; x <= upper[0]; ++x) {
    for (int32_t y = lower[1]; y <= upper[1]; ++y) {
      for (int32_t z = lower[2]; z <= upper[2]; ++z) {
        const double delta[3] = {
            (static_cast<double>(x) + 0.5) * voxel - center[0],
            (static_cast<double>(y) + 0.5) * voxel - center[1],
            (static_cast<double>(z) + 0.5) * voxel - center[2]};
        double value = 0.0;
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            value += delta[i] * inverse[3 * i + j] * delta[j];
        if (value > 1.0 + 1.0e-10) continue;
        visitor(pack_voxel(x, y, z));
        ++visited;
      }
    }
  }
  return visited;
}

MVT_API void* mvt_create(const float* centers, const float* half_extents, int32_t count, float voxel_size) {
  if (!centers || !half_extents || count <= 0 || !(voxel_size > 0.0f)) return nullptr;
  try {
    NativeMVT* m = new NativeMVT();
    m->n = count;
    m->voxel = voxel_size;
    for (int a = 0; a < 3; ++a) {
      m->global_lo[a] = std::numeric_limits<float>::infinity();
      m->global_hi[a] = -std::numeric_limits<float>::infinity();
      m->proxy_lo[a].resize(count);
      m->proxy_hi[a].resize(count);
      for (int i = 0; i < count; ++i) {
        float lo = centers[3*i+a] - half_extents[3*i+a];
        float hi = centers[3*i+a] + half_extents[3*i+a];
        m->proxy_lo[a][i] = lo;
        m->proxy_hi[a][i] = hi;
        m->global_lo[a] = std::min(m->global_lo[a], lo);
        m->global_hi[a] = std::max(m->global_hi[a], hi);
      }
      m->origin[a] = m->global_lo[a] - voxel_size;
      m->dims[a] = std::max(1, static_cast<int32_t>(std::ceil((m->global_hi[a] - m->origin[a]) / voxel_size)) + 1);
    }

    std::unordered_map<int64_t, VoxelBuild> cells;
    for (int i = 0; i < count; ++i) {
      int lo[3], hi[3];
      for (int a = 0; a < 3; ++a) {
        lo[a] = m->clamp_index(m->proxy_lo[a][i] - 1e-6f * voxel_size, a);
        hi[a] = m->clamp_index(m->proxy_hi[a][i] + 1e-6f * voxel_size, a);
      }
      for (int x = lo[0]; x <= hi[0]; ++x)
        for (int y = lo[1]; y <= hi[1]; ++y)
          for (int z = lo[2]; z <= hi[2]; ++z) {
            auto& cell = cells[m->flat(x,y,z)];
            cell.indices.push_back(i);
            for (int a = 0; a < 3; ++a) {
              cell.lo[a] = std::min(cell.lo[a], m->proxy_lo[a][i]);
              cell.hi[a] = std::max(cell.hi[a], m->proxy_hi[a][i]);
            }
          }
    }

    m->x_table.assign(m->dims[0], -1);
    std::unordered_set<int> used_x;
    std::unordered_set<int64_t> used_xy;
    for (const auto& item : cells) {
      int64_t flat = item.first;
      int z = static_cast<int>(flat % m->dims[2]); (void)z;
      int64_t xy = flat / m->dims[2];
      int y = static_cast<int>(xy % m->dims[1]);
      int x = static_cast<int>(xy / m->dims[1]);
      used_x.insert(x);
      used_xy.insert(static_cast<int64_t>(x) * m->dims[1] + y);
    }
    std::vector<int> sorted_x(used_x.begin(), used_x.end());
    std::sort(sorted_x.begin(), sorted_x.end());
    for (int x : sorted_x) {
      int offset = static_cast<int>(m->y_pool.size());
      m->x_table[x] = offset;
      m->y_pool.resize(m->y_pool.size() + m->dims[1], -1);
    }
    std::vector<int64_t> sorted_xy(used_xy.begin(), used_xy.end());
    std::sort(sorted_xy.begin(), sorted_xy.end());
    for (int64_t xy : sorted_xy) {
      int x = static_cast<int>(xy / m->dims[1]);
      int y = static_cast<int>(xy % m->dims[1]);
      int offset = static_cast<int>(m->z_pool.size());
      m->y_pool[m->x_table[x] + y] = offset;
      m->z_pool.resize(m->z_pool.size() + m->dims[2], -1);
    }

    std::vector<int64_t> sorted_cells;
    sorted_cells.reserve(cells.size());
    for (const auto& item : cells) sorted_cells.push_back(item.first);
    std::sort(sorted_cells.begin(), sorted_cells.end());
    for (int64_t flat : sorted_cells) {
      int z = static_cast<int>(flat % m->dims[2]);
      int64_t xy = flat / m->dims[2];
      int y = static_cast<int>(xy % m->dims[1]);
      int x = static_cast<int>(xy / m->dims[1]);
      int voxel_id = static_cast<int>(m->voxel_start.size());
      m->z_pool[m->y_pool[m->x_table[x] + y] + z] = voxel_id;
      auto& cell = cells[flat];
      std::sort(cell.indices.begin(), cell.indices.end());
      cell.indices.erase(std::unique(cell.indices.begin(), cell.indices.end()), cell.indices.end());
      m->voxel_start.push_back(static_cast<int32_t>(m->index_pool.size()));
      m->voxel_count.push_back(static_cast<int32_t>(cell.indices.size()));
      m->index_pool.insert(m->index_pool.end(), cell.indices.begin(), cell.indices.end());
      for (int a = 0; a < 3; ++a) {
        m->voxel_lo[a].push_back(cell.lo[a]);
        m->voxel_hi[a].push_back(cell.hi[a]);
      }
    }
    m->marks.assign(count, 0);
    return m;
  } catch (...) {
    return nullptr;
  }
}

MVT_API void mvt_destroy(void* handle) { delete static_cast<NativeMVT*>(handle); }

static int32_t mvt_query_aabb_impl(void* handle, const float* center, const float* half, int32_t* output, int32_t capacity, bool use_simd) {
  NativeMVT* m = static_cast<NativeMVT*>(handle);
  if (!m || !center || !half || !output || capacity <= 0) return -1;
  float qlo[3], qhi[3];
  for (int a = 0; a < 3; ++a) { qlo[a] = center[a] - half[a]; qhi[a] = center[a] + half[a]; }
  if (!intersects(qlo, qhi, m->global_lo, m->global_hi)) return 0;
  int lo[3], hi[3];
  for (int a = 0; a < 3; ++a) { lo[a] = m->clamp_index(qlo[a] - 1e-6f*m->voxel, a); hi[a] = m->clamp_index(qhi[a] + 1e-6f*m->voxel, a); }
  if (++m->epoch == 0) { std::fill(m->marks.begin(), m->marks.end(), 0); m->epoch = 1; }
  std::vector<int32_t> candidates;
  for (int x = lo[0]; x <= hi[0]; ++x) {
    int yoff = m->x_table[x]; if (yoff < 0) continue;
    for (int y = lo[1]; y <= hi[1]; ++y) {
      int zoff = m->y_pool[yoff+y]; if (zoff < 0) continue;
      for (int z = lo[2]; z <= hi[2]; ++z) {
        int vid = m->z_pool[zoff+z]; if (vid < 0) continue;
        float vlo[3] = {m->voxel_lo[0][vid],m->voxel_lo[1][vid],m->voxel_lo[2][vid]};
        float vhi[3] = {m->voxel_hi[0][vid],m->voxel_hi[1][vid],m->voxel_hi[2][vid]};
        if (!intersects(qlo,qhi,vlo,vhi)) continue;
        int start=m->voxel_start[vid], count=m->voxel_count[vid];
        for (int k=0;k<count;++k) { int idx=m->index_pool[start+k]; if(m->marks[idx]!=m->epoch){m->marks[idx]=m->epoch;candidates.push_back(idx);} }
      }
    }
  }
  int32_t written = 0;
  if (!use_simd) {
    for (int32_t idx : candidates) {
      float plo[3] = {m->proxy_lo[0][idx], m->proxy_lo[1][idx], m->proxy_lo[2][idx]};
      float phi[3] = {m->proxy_hi[0][idx], m->proxy_hi[1][idx], m->proxy_hi[2][idx]};
      if (intersects(qlo, qhi, plo, phi)) {
        if (written >= capacity) return -2;
        output[written++] = idx;
      }
    }
    std::sort(output, output + written);
    return written;
  }
  const __m256 qlox=_mm256_set1_ps(qlo[0]), qloy=_mm256_set1_ps(qlo[1]), qloz=_mm256_set1_ps(qlo[2]);
  const __m256 qhix=_mm256_set1_ps(qhi[0]), qhiy=_mm256_set1_ps(qhi[1]), qhiz=_mm256_set1_ps(qhi[2]);
  size_t k=0;
  alignas(32) int32_t idxbuf[8];
  for (; k+8<=candidates.size(); k+=8) {
    for(int lane=0;lane<8;++lane) idxbuf[lane]=candidates[k+lane];
    __m256i ids=_mm256_load_si256(reinterpret_cast<const __m256i*>(idxbuf));
    __m256 mask=_mm256_cmp_ps(_mm256_i32gather_ps(m->proxy_lo[0].data(),ids,4),qhix,_CMP_LE_OQ);
    mask=_mm256_and_ps(mask,_mm256_cmp_ps(_mm256_i32gather_ps(m->proxy_hi[0].data(),ids,4),qlox,_CMP_GE_OQ));
    mask=_mm256_and_ps(mask,_mm256_cmp_ps(_mm256_i32gather_ps(m->proxy_lo[1].data(),ids,4),qhiy,_CMP_LE_OQ));
    mask=_mm256_and_ps(mask,_mm256_cmp_ps(_mm256_i32gather_ps(m->proxy_hi[1].data(),ids,4),qloy,_CMP_GE_OQ));
    mask=_mm256_and_ps(mask,_mm256_cmp_ps(_mm256_i32gather_ps(m->proxy_lo[2].data(),ids,4),qhiz,_CMP_LE_OQ));
    mask=_mm256_and_ps(mask,_mm256_cmp_ps(_mm256_i32gather_ps(m->proxy_hi[2].data(),ids,4),qloz,_CMP_GE_OQ));
    int bits=_mm256_movemask_ps(mask);
    for(int lane=0;lane<8;++lane) if(bits&(1<<lane)){if(written>=capacity)return -2;output[written++]=idxbuf[lane];}
  }
  for(;k<candidates.size();++k){int idx=candidates[k];float plo[3]={m->proxy_lo[0][idx],m->proxy_lo[1][idx],m->proxy_lo[2][idx]};float phi[3]={m->proxy_hi[0][idx],m->proxy_hi[1][idx],m->proxy_hi[2][idx]};if(intersects(qlo,qhi,plo,phi)){if(written>=capacity)return -2;output[written++]=idx;}}
  std::sort(output, output+written);
  return written;
}

MVT_API int32_t mvt_query_aabb(void* handle, const float* center, const float* half, int32_t* output, int32_t capacity) {
  return mvt_query_aabb_impl(handle, center, half, output, capacity, true);
}

MVT_API int32_t mvt_query_aabb_scalar(void* handle, const float* center, const float* half, int32_t* output, int32_t capacity) {
  return mvt_query_aabb_impl(handle, center, half, output, capacity, false);
}

MVT_API int32_t mvt_proxy_count(void* handle){auto*m=static_cast<NativeMVT*>(handle);return m?m->n:0;}
MVT_API int32_t mvt_occupied_cells(void* handle){auto*m=static_cast<NativeMVT*>(handle);return m?static_cast<int32_t>(m->voxel_start.size()):0;}
MVT_API int64_t mvt_index_references(void* handle){auto*m=static_cast<NativeMVT*>(handle);return m?static_cast<int64_t>(m->index_pool.size()):0;}
MVT_API int32_t mvt_simd_width(){return 8;}

MVT_API void* mvt_multilevel_create(
    const float* centers, const float* half_extents, int32_t count,
    float base_voxel_size, float maximum_query_half_extent) {
  if (!centers || !half_extents || count <= 0 ||
      !(base_voxel_size > 0.0f) || !(maximum_query_half_extent >= 0.0f))
    return nullptr;
  try {
    NativeMultilevelMVT* m = new NativeMultilevelMVT();
    m->n = count;
    m->base_voxel = base_voxel_size;
    m->maximum_query_half = maximum_query_half_extent;
    std::vector<int32_t> proxy_level(count, 0);
    int32_t maximum_level = 0;
    for (int axis = 0; axis < 3; ++axis) {
      m->global_lo[axis] = std::numeric_limits<float>::infinity();
      m->global_hi[axis] = -std::numeric_limits<float>::infinity();
      m->proxy_lo[axis].resize(count);
      m->proxy_hi[axis].resize(count);
    }
    for (int32_t index = 0; index < count; ++index) {
      float maximum_proxy_half = 0.0f;
      for (int axis = 0; axis < 3; ++axis) {
        const float half = half_extents[3 * index + axis];
        if (!(half >= 0.0f) || !std::isfinite(half)) {
          delete m;
          return nullptr;
        }
        const float lo = centers[3 * index + axis] - half;
        const float hi = centers[3 * index + axis] + half;
        m->proxy_lo[axis][index] = lo;
        m->proxy_hi[axis][index] = hi;
        m->global_lo[axis] = std::min(m->global_lo[axis], lo);
        m->global_hi[axis] = std::max(m->global_hi[axis], hi);
        maximum_proxy_half = std::max(maximum_proxy_half, half);
      }
      // Strict slack absorbs float roundoff at exact cell boundaries.
      const float required = maximum_query_half_extent + maximum_proxy_half;
      float voxel = base_voxel_size;
      int32_t level = 0;
      while (!(voxel > required + 8.0f * std::numeric_limits<float>::epsilon() *
                                  std::max(1.0f, required))) {
        voxel *= 2.0f;
        ++level;
        if (level > 30 || !std::isfinite(voxel)) {
          delete m;
          return nullptr;
        }
      }
      proxy_level[index] = level;
      maximum_level = std::max(maximum_level, level);
    }
    m->levels.resize(static_cast<size_t>(maximum_level) + 1);
    for (int32_t level = 0; level <= maximum_level; ++level)
      m->levels[level].voxel = std::ldexp(base_voxel_size, level);

    std::vector<std::unordered_map<uint64_t, VoxelBuild>> build(
        static_cast<size_t>(maximum_level) + 1);
    for (int32_t index = 0; index < count; ++index) {
      const int32_t level_index = proxy_level[index];
      const float voxel = m->levels[level_index].voxel;
      int32_t cell[3];
      for (int axis = 0; axis < 3; ++axis)
        cell[axis] = static_cast<int32_t>(
            std::floor(centers[3 * index + axis] / voxel));
      auto& entry = build[level_index][pack_voxel(cell[0], cell[1], cell[2])];
      entry.indices.push_back(index);
      for (int axis = 0; axis < 3; ++axis) {
        entry.lo[axis] = std::min(entry.lo[axis], m->proxy_lo[axis][index]);
        entry.hi[axis] = std::max(entry.hi[axis], m->proxy_hi[axis][index]);
      }
    }

    for (int32_t level_index = 0; level_index <= maximum_level; ++level_index) {
      auto& level = m->levels[level_index];
      auto& source = build[level_index];
      std::vector<uint64_t> keys;
      keys.reserve(source.size());
      for (const auto& item : source) keys.push_back(item.first);
      std::sort(keys.begin(), keys.end());
      for (const uint64_t key : keys) {
        auto& cell = source[key];
        std::sort(cell.indices.begin(), cell.indices.end());
        const int32_t cell_id = static_cast<int32_t>(level.cell_start.size());
        level.cell_ids.emplace(key, cell_id);
        level.cell_start.push_back(static_cast<int32_t>(level.index_pool.size()));
        level.cell_count.push_back(static_cast<int32_t>(cell.indices.size()));
        for (const int32_t index : cell.indices) {
          level.index_pool.push_back(index);
          for (int axis = 0; axis < 3; ++axis) {
            level.packed_lo[axis].push_back(m->proxy_lo[axis][index]);
            level.packed_hi[axis].push_back(m->proxy_hi[axis][index]);
          }
        }
        for (int axis = 0; axis < 3; ++axis) {
          level.cell_lo[axis].push_back(cell.lo[axis]);
          level.cell_hi[axis].push_back(cell.hi[axis]);
        }
      }
    }
    return m;
  } catch (...) {
    return nullptr;
  }
}

MVT_API void mvt_multilevel_destroy(void* handle) {
  delete static_cast<NativeMultilevelMVT*>(handle);
}

static int32_t mvt_multilevel_query_aabb_impl(
    void* handle, const float* center, const float* half, int32_t* output,
    int32_t capacity, bool use_simd) {
  NativeMultilevelMVT* m = static_cast<NativeMultilevelMVT*>(handle);
  if (!m || !center || !half || !output || capacity <= 0) return -1;
  float qlo[3], qhi[3];
  for (int axis = 0; axis < 3; ++axis) {
    if (!(half[axis] >= 0.0f) ||
        half[axis] > m->maximum_query_half + 2.0e-6f)
      return -3;
    qlo[axis] = center[axis] - half[axis];
    qhi[axis] = center[axis] + half[axis];
  }
  if (!intersects(qlo, qhi, m->global_lo, m->global_hi)) return 0;

  int32_t written = 0;
  const __m256 qlox = _mm256_set1_ps(qlo[0]);
  const __m256 qloy = _mm256_set1_ps(qlo[1]);
  const __m256 qloz = _mm256_set1_ps(qlo[2]);
  const __m256 qhix = _mm256_set1_ps(qhi[0]);
  const __m256 qhiy = _mm256_set1_ps(qhi[1]);
  const __m256 qhiz = _mm256_set1_ps(qhi[2]);
  for (const auto& level : m->levels) {
    int32_t query_cell[3];
    for (int axis = 0; axis < 3; ++axis)
      query_cell[axis] = static_cast<int32_t>(
          std::floor(center[axis] / level.voxel));
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
          const uint64_t key = pack_voxel(
              query_cell[0] + dx, query_cell[1] + dy,
              query_cell[2] + dz);
          const auto found = level.cell_ids.find(key);
          if (found == level.cell_ids.end()) continue;
          const int32_t cell_id = found->second;
          float cell_lo[3], cell_hi[3];
          for (int axis = 0; axis < 3; ++axis) {
            cell_lo[axis] = level.cell_lo[axis][cell_id];
            cell_hi[axis] = level.cell_hi[axis][cell_id];
          }
          if (!intersects(qlo, qhi, cell_lo, cell_hi)) continue;
          const int32_t start = level.cell_start[cell_id];
          const int32_t count = level.cell_count[cell_id];
          int32_t local = 0;
          if (use_simd) {
            for (; local + 8 <= count; local += 8) {
              const int32_t packed = start + local;
              __m256 mask = _mm256_cmp_ps(
                  _mm256_loadu_ps(level.packed_lo[0].data() + packed),
                  qhix, _CMP_LE_OQ);
              mask = _mm256_and_ps(mask, _mm256_cmp_ps(
                  _mm256_loadu_ps(level.packed_hi[0].data() + packed),
                  qlox, _CMP_GE_OQ));
              mask = _mm256_and_ps(mask, _mm256_cmp_ps(
                  _mm256_loadu_ps(level.packed_lo[1].data() + packed),
                  qhiy, _CMP_LE_OQ));
              mask = _mm256_and_ps(mask, _mm256_cmp_ps(
                  _mm256_loadu_ps(level.packed_hi[1].data() + packed),
                  qloy, _CMP_GE_OQ));
              mask = _mm256_and_ps(mask, _mm256_cmp_ps(
                  _mm256_loadu_ps(level.packed_lo[2].data() + packed),
                  qhiz, _CMP_LE_OQ));
              mask = _mm256_and_ps(mask, _mm256_cmp_ps(
                  _mm256_loadu_ps(level.packed_hi[2].data() + packed),
                  qloz, _CMP_GE_OQ));
              const int bits = _mm256_movemask_ps(mask);
              for (int lane = 0; lane < 8; ++lane) {
                if (!(bits & (1 << lane))) continue;
                if (written >= capacity) return -2;
                output[written++] = level.index_pool[packed + lane];
              }
            }
          }
          for (; local < count; ++local) {
            const int32_t packed = start + local;
            const float proxy_lo[3] = {
                level.packed_lo[0][packed],
                level.packed_lo[1][packed],
                level.packed_lo[2][packed]};
            const float proxy_hi[3] = {
                level.packed_hi[0][packed],
                level.packed_hi[1][packed],
                level.packed_hi[2][packed]};
            if (!intersects(qlo, qhi, proxy_lo, proxy_hi)) continue;
            if (written >= capacity) return -2;
            output[written++] = level.index_pool[packed];
          }
        }
  }
  std::sort(output, output + written);
  return written;
}

MVT_API int32_t mvt_multilevel_query_aabb(
    void* handle, const float* center, const float* half, int32_t* output,
    int32_t capacity) {
  return mvt_multilevel_query_aabb_impl(
      handle, center, half, output, capacity, true);
}

MVT_API int32_t mvt_multilevel_query_aabb_scalar(
    void* handle, const float* center, const float* half, int32_t* output,
    int32_t capacity) {
  return mvt_multilevel_query_aabb_impl(
      handle, center, half, output, capacity, false);
}

// Batch form of the exact same 27-cell multilevel query.  ``offsets`` has
// query_count+1 elements and partitions the flat output array.  Each row is
// still independently sorted by the scalar/SIMD implementation above.
MVT_API int32_t mvt_multilevel_query_aabb_batch(
    void* handle, const float* centers, const float* halves,
    int32_t query_count, int32_t use_simd, int32_t* offsets,
    int32_t* output, int32_t capacity) {
  if (!handle || !centers || !halves || !offsets || !output ||
      query_count < 0 || capacity <= 0) return -1;
  int32_t written = 0;
  offsets[0] = 0;
  for (int32_t query = 0; query < query_count; ++query) {
    const int32_t count = mvt_multilevel_query_aabb_impl(
        handle, centers + 3*query, halves + 3*query, output + written,
        capacity - written, use_simd != 0);
    if (count < 0) return count;
    written += count;
    offsets[query + 1] = written;
  }
  return written;
}

static float point_aabb_distance_squared(
    const float* point, const float* lo, const float* hi) {
  float squared = 0.0f;
  for (int axis = 0; axis < 3; ++axis) {
    const float delta = point[axis] < lo[axis]
        ? lo[axis] - point[axis]
        : (point[axis] > hi[axis] ? point[axis] - hi[axis] : 0.0f);
    squared += delta * delta;
  }
  return squared;
}

// VCC-style ball--AABB narrow broadphase.  The same multilevel table and
// exactly 27 cell probes per populated level are retained, but cube-corner
// false positives from the historical AABB--AABB query are rejected by the
// exact Euclidean point-to-box distance predicate.  Since every proxy
// certificate is contained in its stored AABB, this cannot remove a proxy
// whose AABB intersects the padded robot sphere.
static int32_t mvt_multilevel_query_sphere_impl(
    void* handle, const float* center, float radius, int32_t* output,
    int32_t capacity, bool use_simd, bool sort_output) {
  NativeMultilevelMVT* m = static_cast<NativeMultilevelMVT*>(handle);
  if (!m || !center || !output || capacity <= 0) return -1;
  if (!(radius >= 0.0f) ||
      radius > m->maximum_query_half + 2.0e-6f) return -3;
  const float radius_squared = radius * radius;
  float qlo[3], qhi[3];
  for (int axis = 0; axis < 3; ++axis) {
    qlo[axis] = center[axis] - radius;
    qhi[axis] = center[axis] + radius;
  }
  if (!intersects(qlo, qhi, m->global_lo, m->global_hi)) return 0;

  int32_t written = 0;
  const __m256 cx = _mm256_set1_ps(center[0]);
  const __m256 cy = _mm256_set1_ps(center[1]);
  const __m256 cz = _mm256_set1_ps(center[2]);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 radius2 = _mm256_set1_ps(radius_squared);
  for (const auto& level : m->levels) {
    int32_t query_cell[3];
    for (int axis = 0; axis < 3; ++axis)
      query_cell[axis] = static_cast<int32_t>(
          std::floor(center[axis] / level.voxel));
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
          const uint64_t key = pack_voxel(
              query_cell[0] + dx, query_cell[1] + dy,
              query_cell[2] + dz);
          const auto found = level.cell_ids.find(key);
          if (found == level.cell_ids.end()) continue;
          const int32_t cell_id = found->second;
          float cell_lo[3], cell_hi[3];
          for (int axis = 0; axis < 3; ++axis) {
            cell_lo[axis] = level.cell_lo[axis][cell_id];
            cell_hi[axis] = level.cell_hi[axis][cell_id];
          }
          if (point_aabb_distance_squared(center, cell_lo, cell_hi) >
              radius_squared) continue;
          const int32_t start = level.cell_start[cell_id];
          const int32_t count = level.cell_count[cell_id];
          int32_t local = 0;
          if (use_simd) {
            for (; local + 8 <= count; local += 8) {
              const int32_t packed = start + local;
              const __m256 lox = _mm256_loadu_ps(
                  level.packed_lo[0].data() + packed);
              const __m256 hix = _mm256_loadu_ps(
                  level.packed_hi[0].data() + packed);
              const __m256 loy = _mm256_loadu_ps(
                  level.packed_lo[1].data() + packed);
              const __m256 hiy = _mm256_loadu_ps(
                  level.packed_hi[1].data() + packed);
              const __m256 loz = _mm256_loadu_ps(
                  level.packed_lo[2].data() + packed);
              const __m256 hiz = _mm256_loadu_ps(
                  level.packed_hi[2].data() + packed);
              const __m256 ddx = _mm256_max_ps(
                  _mm256_max_ps(_mm256_sub_ps(lox, cx),
                                _mm256_sub_ps(cx, hix)), zero);
              const __m256 ddy = _mm256_max_ps(
                  _mm256_max_ps(_mm256_sub_ps(loy, cy),
                                _mm256_sub_ps(cy, hiy)), zero);
              const __m256 ddz = _mm256_max_ps(
                  _mm256_max_ps(_mm256_sub_ps(loz, cz),
                                _mm256_sub_ps(cz, hiz)), zero);
              __m256 squared = _mm256_mul_ps(ddx, ddx);
              squared = _mm256_add_ps(
                  squared, _mm256_mul_ps(ddy, ddy));
              squared = _mm256_add_ps(
                  squared, _mm256_mul_ps(ddz, ddz));
              const int bits = _mm256_movemask_ps(_mm256_cmp_ps(
                  squared, radius2, _CMP_LE_OQ));
              for (int lane = 0; lane < 8; ++lane) {
                if (!(bits & (1 << lane))) continue;
                if (written >= capacity) return -2;
                output[written++] = level.index_pool[packed + lane];
              }
            }
          }
          for (; local < count; ++local) {
            const int32_t packed = start + local;
            const float proxy_lo[3] = {
                level.packed_lo[0][packed],
                level.packed_lo[1][packed],
                level.packed_lo[2][packed]};
            const float proxy_hi[3] = {
                level.packed_hi[0][packed],
                level.packed_hi[1][packed],
                level.packed_hi[2][packed]};
            if (point_aabb_distance_squared(center, proxy_lo, proxy_hi) >
                radius_squared) continue;
            if (written >= capacity) return -2;
            output[written++] = level.index_pool[packed];
          }
        }
  }
  if (sort_output) std::sort(output, output + written);
  return written;
}

MVT_API int32_t mvt_multilevel_query_sphere(
    void* handle, const float* center, float radius, int32_t* output,
    int32_t capacity, int32_t use_simd) {
  return mvt_multilevel_query_sphere_impl(
      handle, center, radius, output, capacity, use_simd != 0, true);
}

MVT_API int32_t mvt_multilevel_query_sphere_batch(
    void* handle, const float* centers, const float* radii,
    int32_t query_count, int32_t use_simd, int32_t* offsets,
    int32_t* output, int32_t capacity) {
  if (!handle || !centers || !radii || !offsets || !output ||
      query_count < 0 || capacity <= 0) return -1;
  int32_t written = 0;
  offsets[0] = 0;
  for (int32_t query = 0; query < query_count; ++query) {
    const int32_t count = mvt_multilevel_query_sphere_impl(
        handle, centers + 3*query, radii[query], output + written,
        capacity - written, use_simd != 0, true);
    if (count < 0) return count;
    written += count;
    offsets[query + 1] = written;
  }
  return written;
}

// Controller-only batch form.  Each proxy belongs to exactly one MVT level
// and one cell, so the 27-cell traversal is already unique.  The downstream
// exact kernel immediately applies its distance-key ordering; skipping this
// proxy-ID ordering removes a provably redundant O(N log N) pass.
MVT_API int32_t mvt_multilevel_query_sphere_batch_unordered(
    void* handle, const float* centers, const float* radii,
    int32_t query_count, int32_t use_simd, int32_t* offsets,
    int32_t* output, int32_t capacity) {
  if (!handle || !centers || !radii || !offsets || !output ||
      query_count < 0 || capacity <= 0) return -1;
  int32_t written = 0;
  offsets[0] = 0;
  for (int32_t query = 0; query < query_count; ++query) {
    const int32_t count = mvt_multilevel_query_sphere_impl(
        handle, centers + 3*query, radii[query], output + written,
        capacity - written, use_simd != 0, false);
    if (count < 0) return count;
    written += count;
    offsets[query + 1] = written;
  }
  return written;
}

MVT_API int32_t mvt_multilevel_proxy_count(void* handle) {
  auto* m = static_cast<NativeMultilevelMVT*>(handle);
  return m ? m->n : 0;
}
MVT_API int32_t mvt_multilevel_level_count(void* handle) {
  auto* m = static_cast<NativeMultilevelMVT*>(handle);
  return m ? static_cast<int32_t>(m->levels.size()) : 0;
}
MVT_API int32_t mvt_multilevel_occupied_cells(void* handle) {
  auto* m = static_cast<NativeMultilevelMVT*>(handle);
  if (!m) return 0;
  int32_t count = 0;
  for (const auto& level : m->levels)
    count += static_cast<int32_t>(level.cell_start.size());
  return count;
}
MVT_API int64_t mvt_multilevel_index_references(void* handle) {
  auto* m = static_cast<NativeMultilevelMVT*>(handle);
  if (!m) return 0;
  int64_t count = 0;
  for (const auto& level : m->levels)
    count += static_cast<int64_t>(level.index_pool.size());
  return count;
}
MVT_API int32_t mvt_multilevel_level_proxy_count(void* handle, int32_t index) {
  auto* m = static_cast<NativeMultilevelMVT*>(handle);
  if (!m || index < 0 || index >= static_cast<int32_t>(m->levels.size()))
    return -1;
  return static_cast<int32_t>(m->levels[index].index_pool.size());
}
MVT_API float mvt_multilevel_level_voxel_size(void* handle, int32_t index) {
  auto* m = static_cast<NativeMultilevelMVT*>(handle);
  if (!m || index < 0 || index >= static_cast<int32_t>(m->levels.size()))
    return -1.0f;
  return m->levels[index].voxel;
}

static double support3(const double* shape, const double* normal) {
  const double x = shape[0]*normal[0] + shape[1]*normal[1] + shape[2]*normal[2];
  const double y = shape[3]*normal[0] + shape[4]*normal[1] + shape[5]*normal[2];
  const double z = shape[6]*normal[0] + shape[7]*normal[1] + shape[8]*normal[2];
  return std::sqrt(std::max(0.0, normal[0]*x + normal[1]*y + normal[2]*z));
}

static double support_objective3(
    const double* delta, const double* robot_shape,
    const double* obstacle_shape, const double* uncertainty_shape,
    const double* normal) {
  return normal[0]*delta[0] + normal[1]*delta[1] + normal[2]*delta[2]
      - support3(robot_shape, normal) - support3(obstacle_shape, normal)
      - (uncertainty_shape ? support3(uncertainty_shape, normal) : 0.0);
}

// Batch implementation of the same deterministic projected-ascent support
// normal used by ellipsoid_model.optimal_support_separating_normal.  Keeping
// it in the native library removes Python call/array overhead without changing
// the mathematical objective or accepted backtracking steps.
static int32_t ellipsoid_support_normals_impl(
    const double* robot_center, const double* robot_shape,
    const double* obstacle_centers, const double* obstacle_shapes,
    const double* uncertainty_shapes,
    int32_t count, int32_t max_iterations, double* output_normals,
    int32_t* output_iterations) {
  if (!robot_center || !robot_shape || !obstacle_centers || !obstacle_shapes ||
      !output_normals || count < 0 || max_iterations < 0) return -1;
  constexpr double steps[5] = {0.5, 0.25, 0.125, 0.0625, 0.03125};
  for (int32_t item = 0; item < count; ++item) {
    const double* obstacle_center = obstacle_centers + 3*item;
    const double* obstacle_shape = obstacle_shapes + 9*item;
    const double* uncertainty_shape = uncertainty_shapes ? uncertainty_shapes + 9*item : nullptr;
    double delta[3] = {
        obstacle_center[0]-robot_center[0],
        obstacle_center[1]-robot_center[1],
        obstacle_center[2]-robot_center[2]};
    const double distance = std::sqrt(
        delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
    double normal[3] = {1.0, 0.0, 0.0};
    if (distance > 1.0e-12) {
      normal[0]=delta[0]/distance; normal[1]=delta[1]/distance; normal[2]=delta[2]/distance;
    }
    double value = support_objective3(
        delta, robot_shape, obstacle_shape, uncertainty_shape, normal);
    int32_t iterations = 0;
    for (; iterations < max_iterations; ++iterations) {
      const double robot_extent = std::max(support3(robot_shape, normal), 1.0e-12);
      const double obstacle_extent = std::max(support3(obstacle_shape, normal), 1.0e-12);
      const double uncertainty_extent = uncertainty_shape
          ? std::max(support3(uncertainty_shape, normal), 1.0e-12) : 1.0;
      double qr[3], qo[3], qu[3] = {0.0, 0.0, 0.0};
      for (int row=0; row<3; ++row) {
        qr[row] = robot_shape[3*row]*normal[0] + robot_shape[3*row+1]*normal[1] + robot_shape[3*row+2]*normal[2];
        qo[row] = obstacle_shape[3*row]*normal[0] + obstacle_shape[3*row+1]*normal[1] + obstacle_shape[3*row+2]*normal[2];
        if (uncertainty_shape) {
          qu[row] = uncertainty_shape[3*row]*normal[0]
              + uncertainty_shape[3*row+1]*normal[1]
              + uncertainty_shape[3*row+2]*normal[2];
        }
      }
      double gradient[3] = {
          delta[0]-qr[0]/robot_extent-qo[0]/obstacle_extent-qu[0]/uncertainty_extent,
          delta[1]-qr[1]/robot_extent-qo[1]/obstacle_extent-qu[1]/uncertainty_extent,
          delta[2]-qr[2]/robot_extent-qo[2]/obstacle_extent-qu[2]/uncertainty_extent};
      const double radial = normal[0]*gradient[0]+normal[1]*gradient[1]+normal[2]*gradient[2];
      double tangent[3] = {
          gradient[0]-normal[0]*radial,
          gradient[1]-normal[1]*radial,
          gradient[2]-normal[2]*radial};
      const double tangent_norm = std::sqrt(tangent[0]*tangent[0]+tangent[1]*tangent[1]+tangent[2]*tangent[2]);
      if (tangent_norm <= 1.0e-10) break;
      for (double& component : tangent) component /= tangent_norm;
      bool improved = false;
      for (double step : steps) {
        double candidate[3] = {
            normal[0]+step*tangent[0], normal[1]+step*tangent[1], normal[2]+step*tangent[2]};
        const double norm = std::sqrt(candidate[0]*candidate[0]+candidate[1]*candidate[1]+candidate[2]*candidate[2]);
        for (double& component : candidate) component /= norm;
        const double candidate_value = support_objective3(
            delta, robot_shape, obstacle_shape, uncertainty_shape, candidate);
        if (candidate_value > value + 1.0e-12) {
          normal[0]=candidate[0]; normal[1]=candidate[1]; normal[2]=candidate[2];
          value=candidate_value; improved=true; break;
        }
      }
      if (!improved) break;
    }
    output_normals[3*item]=normal[0]; output_normals[3*item+1]=normal[1]; output_normals[3*item+2]=normal[2];
    if (output_iterations) output_iterations[item]=iterations;
  }
  return count;
}

MVT_API int32_t ellipsoid_support_normals(
    const double* robot_center, const double* robot_shape,
    const double* obstacle_centers, const double* obstacle_shapes,
    int32_t count, int32_t max_iterations, double* output_normals,
    int32_t* output_iterations) {
  return ellipsoid_support_normals_impl(
      robot_center, robot_shape, obstacle_centers, obstacle_shapes, nullptr,
      count, max_iterations, output_normals, output_iterations);
}

// Exact support-normal optimizer for R (+) B (+) U.  U is the directional
// depth/pixel uncertainty ellipsoid and is deliberately not collapsed into a
// larger single ellipsoid, because doing so can close a physically open slit.
MVT_API int32_t ellipsoid_support_normals_sum(
    const double* robot_center, const double* robot_shape,
    const double* obstacle_centers, const double* obstacle_shapes,
    const double* uncertainty_shapes, int32_t count, int32_t max_iterations,
    double* output_normals, int32_t* output_iterations) {
  if (!uncertainty_shapes) return -1;
  return ellipsoid_support_normals_impl(
      robot_center, robot_shape, obstacle_centers, obstacle_shapes,
      uncertainty_shapes, count, max_iterations, output_normals,
      output_iterations);
}

// Warm-started high-accuracy support optimizer for the formal online
// Q_R+Q_O+U controller.  The older fixed-step batch entry point above is kept
// byte-for-byte compatible with historical benchmarks.  This entry point uses
// Armijo backtracking down to machine-scale angular steps and reports the
// tangent KKT residual for every pair.
MVT_API int32_t ellipsoid_support_normals_sum_warm(
    const double* robot_center, const double* robot_shape,
    const double* obstacle_centers, const double* obstacle_shapes,
    const double* uncertainty_shapes, const double* initial_normals,
    int32_t count, int32_t max_iterations, double* output_normals,
    int32_t* output_iterations, double* output_residuals) {
  if (!robot_center || !robot_shape || !obstacle_centers || !obstacle_shapes ||
      !uncertainty_shapes || !output_normals || !output_iterations ||
      !output_residuals || count < 0 || max_iterations < 0) return -1;
  for (int32_t item = 0; item < count; ++item) {
    const double* center = obstacle_centers + 3*item;
    const double* obstacle = obstacle_shapes + 9*item;
    const double* uncertainty = uncertainty_shapes + 9*item;
    double delta[3] = {
        center[0]-robot_center[0], center[1]-robot_center[1],
        center[2]-robot_center[2]};
    double normal[3] = {delta[0], delta[1], delta[2]};
    if (initial_normals) {
      const double* initial = initial_normals + 3*item;
      const double norm = std::sqrt(initial[0]*initial[0] +
          initial[1]*initial[1] + initial[2]*initial[2]);
      if (std::isfinite(norm) && norm > 1.0e-12) {
        normal[0]=initial[0]/norm; normal[1]=initial[1]/norm;
        normal[2]=initial[2]/norm;
      }
    }
    double norm = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1] +
                            normal[2]*normal[2]);
    if (!(norm > 1.0e-12)) {
      normal[0]=1.0; normal[1]=0.0; normal[2]=0.0;
    } else {
      normal[0]/=norm; normal[1]/=norm; normal[2]/=norm;
    }
    auto objective = [&](const double* direction) {
      return direction[0]*delta[0] + direction[1]*delta[1] +
          direction[2]*delta[2] - support3(robot_shape, direction) -
          support3(obstacle, direction) - support3(uncertainty, direction);
    };
    double value = objective(normal);
    double residual = 0.0;
    int32_t iterations = 0;
    for (; iterations < max_iterations; ++iterations) {
      const double robot_extent = std::max(support3(robot_shape, normal), 1.0e-12);
      const double obstacle_extent = std::max(support3(obstacle, normal), 1.0e-12);
      const double uncertainty_extent = std::max(support3(uncertainty, normal), 1.0e-12);
      double gradient[3] = {delta[0], delta[1], delta[2]};
      for (int row=0; row<3; ++row) {
        const double qr = robot_shape[3*row]*normal[0] +
            robot_shape[3*row+1]*normal[1] + robot_shape[3*row+2]*normal[2];
        const double qo = obstacle[3*row]*normal[0] +
            obstacle[3*row+1]*normal[1] + obstacle[3*row+2]*normal[2];
        const double qu = uncertainty[3*row]*normal[0] +
            uncertainty[3*row+1]*normal[1] + uncertainty[3*row+2]*normal[2];
        gradient[row] -= qr/robot_extent + qo/obstacle_extent +
            qu/uncertainty_extent;
      }
      const double radial = normal[0]*gradient[0] + normal[1]*gradient[1] +
          normal[2]*gradient[2];
      double tangent[3] = {gradient[0]-normal[0]*radial,
          gradient[1]-normal[1]*radial, gradient[2]-normal[2]*radial};
      residual = std::sqrt(tangent[0]*tangent[0] + tangent[1]*tangent[1] +
                           tangent[2]*tangent[2]);
      if (residual <= 1.0e-10) break;
      tangent[0]/=residual; tangent[1]/=residual; tangent[2]/=residual;
      double step = 0.5;
      bool accepted = false;
      for (int trial=0; trial<48; ++trial, step*=0.5) {
        double candidate[3] = {normal[0]+step*tangent[0],
            normal[1]+step*tangent[1], normal[2]+step*tangent[2]};
        const double candidate_norm = std::sqrt(candidate[0]*candidate[0] +
            candidate[1]*candidate[1] + candidate[2]*candidate[2]);
        candidate[0]/=candidate_norm; candidate[1]/=candidate_norm;
        candidate[2]/=candidate_norm;
        const double candidate_value = objective(candidate);
        if (candidate_value >= value + 1.0e-4*step*residual) {
          normal[0]=candidate[0]; normal[1]=candidate[1]; normal[2]=candidate[2];
          value=candidate_value; accepted=true; break;
        }
      }
      if (!accepted) break;
    }
    output_normals[3*item]=normal[0]; output_normals[3*item+1]=normal[1];
    output_normals[3*item+2]=normal[2];
    output_iterations[item]=iterations;
    output_residuals[item]=residual;
  }
  return count;
}

// Safeguarded Riemannian Newton solve of the same exact support objective
//   n.d - sqrt(n'Qr n) - sqrt(n'Qo n) - sqrt(n'U n), ||n||=1.
// The 2x2 Hessian is formed in an orthonormal tangent basis.  An indefinite,
// singular, or non-ascent Newton direction falls back to the Riemannian
// gradient, and every accepted step must pass Armijo backtracking.  Thus this
// changes only the numerical solver, not the support formula or candidate set.
static void ellipsoid_support_normal_sum_newton_one(
    const double* robot_center, const double* robot_shape,
    const double* obstacle_center, const double* obstacle_shape,
    const double* uncertainty_shape, const double* initial_normal,
    int32_t max_iterations, double* output_normal,
    int32_t* output_iterations, double* output_residual) {
  const double delta[3] = {
      obstacle_center[0]-robot_center[0],
      obstacle_center[1]-robot_center[1],
      obstacle_center[2]-robot_center[2]};
  double normal[3] = {delta[0], delta[1], delta[2]};
  if (initial_normal) {
    const double initial_norm = std::sqrt(
        initial_normal[0]*initial_normal[0] +
        initial_normal[1]*initial_normal[1] +
        initial_normal[2]*initial_normal[2]);
    if (std::isfinite(initial_norm) && initial_norm > 1.0e-12) {
      normal[0]=initial_normal[0]/initial_norm;
      normal[1]=initial_normal[1]/initial_norm;
      normal[2]=initial_normal[2]/initial_norm;
    }
  }
  double normal_norm = std::sqrt(
      normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
  if (!(normal_norm > 1.0e-12)) {
    normal[0]=1.0; normal[1]=0.0; normal[2]=0.0;
  } else {
    // A zero-filled warm-start row means "no cached normal".  The caller
    // still passes a non-null batch pointer, so conditioning normalization on
    // pointer nullness starts Newton away from S^2.  Every valid start,
    // whether it came from the cache or the center line, must lie on S^2.
    normal[0]/=normal_norm; normal[1]/=normal_norm; normal[2]/=normal_norm;
  }

  auto derivatives = [&](const double* direction, double* gradient,
                         double* hessian) {
    gradient[0]=delta[0]; gradient[1]=delta[1]; gradient[2]=delta[2];
    for (int index=0; index<9; ++index) hessian[index]=0.0;
    const double* shapes[3] = {
        robot_shape, obstacle_shape, uncertainty_shape};
    for (const double* shape : shapes) {
      double qn[3]{};
      for (int row=0; row<3; ++row) {
        qn[row] = shape[3*row]*direction[0] +
            shape[3*row+1]*direction[1] +
            shape[3*row+2]*direction[2];
      }
      const double extent = std::max(std::sqrt(std::max(
          0.0, direction[0]*qn[0] + direction[1]*qn[1] +
          direction[2]*qn[2])), 1.0e-12);
      for (int row=0; row<3; ++row) gradient[row] -= qn[row]/extent;
      const double inverse_extent = 1.0/extent;
      const double inverse_extent_cubed =
          inverse_extent*inverse_extent*inverse_extent;
      for (int row=0; row<3; ++row)
        for (int column=0; column<3; ++column)
          hessian[3*row+column] +=
              -shape[3*row+column]*inverse_extent +
              qn[row]*qn[column]*inverse_extent_cubed;
    }
  };
  auto objective = [&](const double* direction) {
    return support_objective3(
        delta, robot_shape, obstacle_shape, uncertainty_shape, direction);
  };

  double value = objective(normal);
  double residual = std::numeric_limits<double>::infinity();
  int32_t iterations = 0;
  for (; iterations < max_iterations; ++iterations) {
    double gradient[3]{}, hessian[9]{};
    derivatives(normal, gradient, hessian);
    const double multiplier = normal[0]*gradient[0] +
        normal[1]*gradient[1] + normal[2]*gradient[2];
    double tangent_gradient[3] = {
        gradient[0]-multiplier*normal[0],
        gradient[1]-multiplier*normal[1],
        gradient[2]-multiplier*normal[2]};
    residual = std::sqrt(
        tangent_gradient[0]*tangent_gradient[0] +
        tangent_gradient[1]*tangent_gradient[1] +
        tangent_gradient[2]*tangent_gradient[2]);
    if (residual <= 1.0e-10) break;

    int reference_axis = 0;
    if (std::abs(normal[1]) < std::abs(normal[reference_axis])) reference_axis=1;
    if (std::abs(normal[2]) < std::abs(normal[reference_axis])) reference_axis=2;
    double reference[3] = {0.0, 0.0, 0.0};
    reference[reference_axis]=1.0;
    double tangent1[3] = {
        normal[1]*reference[2]-normal[2]*reference[1],
        normal[2]*reference[0]-normal[0]*reference[2],
        normal[0]*reference[1]-normal[1]*reference[0]};
    const double tangent1_norm = std::sqrt(
        tangent1[0]*tangent1[0] + tangent1[1]*tangent1[1] +
        tangent1[2]*tangent1[2]);
    for (double& component : tangent1) component/=tangent1_norm;
    const double tangent2[3] = {
        normal[1]*tangent1[2]-normal[2]*tangent1[1],
        normal[2]*tangent1[0]-normal[0]*tangent1[2],
        normal[0]*tangent1[1]-normal[1]*tangent1[0]};
    auto hessian_product = [&](const double* vector, double* output) {
      for (int row=0; row<3; ++row) {
        output[row] = hessian[3*row]*vector[0] +
            hessian[3*row+1]*vector[1] +
            hessian[3*row+2]*vector[2] - multiplier*vector[row];
      }
    };
    double ht1[3]{}, ht2[3]{};
    hessian_product(tangent1, ht1);
    hessian_product(tangent2, ht2);
    const double a11=tangent1[0]*ht1[0]+tangent1[1]*ht1[1]+tangent1[2]*ht1[2];
    const double a12=tangent1[0]*ht2[0]+tangent1[1]*ht2[1]+tangent1[2]*ht2[2];
    const double a22=tangent2[0]*ht2[0]+tangent2[1]*ht2[1]+tangent2[2]*ht2[2];
    const double determinant=a11*a22-a12*a12;
    const double r1=tangent1[0]*tangent_gradient[0]+
        tangent1[1]*tangent_gradient[1]+tangent1[2]*tangent_gradient[2];
    const double r2=tangent2[0]*tangent_gradient[0]+
        tangent2[1]*tangent_gradient[1]+tangent2[2]*tangent_gradient[2];
    double direction[3]{};
    bool newton = a11 < -1.0e-12 && a22 < -1.0e-12 &&
        determinant > 1.0e-20 && std::isfinite(determinant);
    if (newton) {
      const double coefficient1=(-a22*r1+a12*r2)/determinant;
      const double coefficient2=(a12*r1-a11*r2)/determinant;
      for (int axis=0; axis<3; ++axis)
        direction[axis]=coefficient1*tangent1[axis]+
            coefficient2*tangent2[axis];
    } else {
      for (int axis=0; axis<3; ++axis)
        direction[axis]=tangent_gradient[axis]/residual;
    }
    double derivative = tangent_gradient[0]*direction[0] +
        tangent_gradient[1]*direction[1] + tangent_gradient[2]*direction[2];
    double direction_norm = std::sqrt(direction[0]*direction[0] +
        direction[1]*direction[1] + direction[2]*direction[2]);
    if (!std::isfinite(derivative) || derivative <= 1.0e-16 ||
        !std::isfinite(direction_norm)) {
      newton=false;
      for (int axis=0; axis<3; ++axis)
        direction[axis]=tangent_gradient[axis]/residual;
      derivative=residual;
      direction_norm=1.0;
    }
    if (direction_norm > 1.0) {
      for (double& component : direction) component/=direction_norm;
      derivative/=direction_norm;
    }

    auto line_search = [&](double initial_step) {
      double step=initial_step;
      for (int trial=0; trial<48; ++trial, step*=0.5) {
        double candidate[3] = {
            normal[0]+step*direction[0],
            normal[1]+step*direction[1],
            normal[2]+step*direction[2]};
        const double candidate_norm=std::sqrt(candidate[0]*candidate[0] +
            candidate[1]*candidate[1]+candidate[2]*candidate[2]);
        for (double& component : candidate) component/=candidate_norm;
        const double candidate_value=objective(candidate);
        if (candidate_value >= value + 1.0e-4*step*derivative) {
          normal[0]=candidate[0]; normal[1]=candidate[1];
          normal[2]=candidate[2]; value=candidate_value;
          return true;
        }
      }
      return false;
    };
    bool accepted=line_search(newton ? 1.0 : 0.5);
    if (!accepted && newton) {
      for (int axis=0; axis<3; ++axis)
        direction[axis]=tangent_gradient[axis]/residual;
      derivative=residual;
      accepted=line_search(0.5);
    }
    if (!accepted) break;
  }
  double final_gradient[3]{}, final_hessian[9]{};
  derivatives(normal, final_gradient, final_hessian);
  const double final_multiplier = normal[0]*final_gradient[0] +
      normal[1]*final_gradient[1] + normal[2]*final_gradient[2];
  residual=std::sqrt(
      std::pow(final_gradient[0]-final_multiplier*normal[0],2) +
      std::pow(final_gradient[1]-final_multiplier*normal[1],2) +
      std::pow(final_gradient[2]-final_multiplier*normal[2],2));
  output_normal[0]=normal[0]; output_normal[1]=normal[1];
  output_normal[2]=normal[2];
  *output_iterations=iterations;
  *output_residual=residual;
}

MVT_API int32_t ellipsoid_support_normals_sum_newton_warm(
    const double* robot_center, const double* robot_shape,
    const double* obstacle_centers, const double* obstacle_shapes,
    const double* uncertainty_shapes, const double* initial_normals,
    int32_t count, int32_t max_iterations, double* output_normals,
    int32_t* output_iterations, double* output_residuals) {
  if (!robot_center || !robot_shape || !obstacle_centers || !obstacle_shapes ||
      !uncertainty_shapes || !output_normals || !output_iterations ||
      !output_residuals || count < 0 || max_iterations < 0) return -1;
  for (int32_t item=0; item<count; ++item)
    ellipsoid_support_normal_sum_newton_one(
        robot_center, robot_shape, obstacle_centers+3*item,
        obstacle_shapes+9*item, uncertainty_shapes+9*item,
        initial_normals ? initial_normals+3*item : nullptr,
        max_iterations, output_normals+3*item, output_iterations+item,
        output_residuals+item);
  return count;
}

MVT_API int32_t ellipsoid_support_normals_sum_pairs_newton_warm(
    const double* robot_centers, const double* robot_shapes,
    const double* obstacle_centers, const double* obstacle_shapes,
    const double* uncertainty_shapes, const double* initial_normals,
    int32_t count, int32_t max_iterations, double* output_normals,
    int32_t* output_iterations, double* output_residuals) {
  if (!robot_centers || !robot_shapes || !obstacle_centers ||
      !obstacle_shapes || !uncertainty_shapes || !output_normals ||
      !output_iterations || !output_residuals || count < 0 ||
      max_iterations < 0) return -1;
#pragma omp parallel for schedule(static) num_threads(4) if(count >= 64)
  for (int32_t item=0; item<count; ++item)
    ellipsoid_support_normal_sum_newton_one(
        robot_centers+3*item, robot_shapes+9*item,
        obstacle_centers+3*item, obstacle_shapes+9*item,
        uncertainty_shapes+9*item,
        initial_normals ? initial_normals+3*item : nullptr,
        max_iterations, output_normals+3*item, output_iterations+item,
        output_residuals+item);
  return count;
}

// Pairwise batch wrapper for swept-volume certificates.  Unlike the formal
// controller batch above, every item may have a different robot center and
// robot shape.  It deliberately calls the exact same warm support solver for
// each item; the only change is collapsing many Python/ctypes calls into one
// native entry point.
MVT_API int32_t ellipsoid_support_normals_sum_pairs_warm(
    const double* robot_centers, const double* robot_shapes,
    const double* obstacle_centers, const double* obstacle_shapes,
    const double* uncertainty_shapes, const double* initial_normals,
    int32_t count, int32_t max_iterations, double* output_normals,
    int32_t* output_iterations, double* output_residuals) {
  if (!robot_centers || !robot_shapes || !obstacle_centers ||
      !obstacle_shapes || !uncertainty_shapes || !output_normals ||
      !output_iterations || !output_residuals || count < 0 ||
      max_iterations < 0) return -1;
  int32_t error_code = 0;
#pragma omp parallel for schedule(static) num_threads(4) if(count >= 64)
  for (int32_t item = 0; item < count; ++item) {
    const int32_t written = ellipsoid_support_normals_sum_warm(
        robot_centers + 3*item, robot_shapes + 9*item,
        obstacle_centers + 3*item, obstacle_shapes + 9*item,
        uncertainty_shapes + 9*item,
        initial_normals ? initial_normals + 3*item : nullptr,
        1, max_iterations, output_normals + 3*item,
        output_iterations + item, output_residuals + item);
    if (written != 1) {
#pragma omp critical(ellipsoid_support_pair_error)
      {
        if (error_code == 0) error_code = written;
      }
    }
  }
  return error_code == 0 ? count : error_code;
}

// Batched form of ellipsoid_model.closest_point_on_ellipsoid.  Each item
// solves the exact one-dimensional KKT root with safeguarded Newton, falls
// back to bracket bisection when needed, and accepts the previous control
// cycle's multiplier as a warm start.  Each KKT root remains the exact scalar
// safeguarded-Newton problem; independent obstacle pairs are parallelized.
// AVX2 remains confined to the MVT/AABB broadphase.
MVT_API int32_t ellipsoid_closest_points_warm(
    const double* point, const double* obstacle_centers,
    const double* eigenvalues, const double* rotations,
    const double* initial_multipliers, int32_t count,
    int32_t max_newton_iterations, int32_t max_bisection_iterations,
    double tolerance, double* output_normals, double* output_surfaces,
    double* output_multipliers, int32_t* output_newton_iterations,
    int32_t* output_bisection_iterations, double* output_residuals) {
  if (!point || !obstacle_centers || !eigenvalues || !rotations ||
      !output_normals || !output_surfaces || !output_multipliers ||
      !output_newton_iterations || !output_bisection_iterations ||
      !output_residuals || count < 0 || max_newton_iterations < 0 ||
      max_bisection_iterations < 0 || !(tolerance > 0.0)) return -1;
#pragma omp parallel for schedule(static) num_threads(4) if(count >= 64)
  for (int32_t item = 0; item < count; ++item) {
    const double* center = obstacle_centers + 3*item;
    const double* values_in = eigenvalues + 3*item;
    const double* rotation = rotations + 9*item;
    double values[3] = {
        std::max(values_in[0], 1.0e-18),
        std::max(values_in[1], 1.0e-18),
        std::max(values_in[2], 1.0e-18)};
    const double delta[3] = {
        point[0]-center[0], point[1]-center[1], point[2]-center[2]};
    double local[3]{};
    for (int axis=0; axis<3; ++axis) {
      local[axis] = rotation[axis]*delta[0] +
          rotation[3+axis]*delta[1] + rotation[6+axis]*delta[2];
    }
    const double inside = local[0]*local[0]/values[0] +
        local[1]*local[1]/values[1] + local[2]*local[2]/values[2];
    double* normal = output_normals + 3*item;
    double* surface = output_surfaces + 3*item;
    if (inside <= 1.0 + 1.0e-12) {
      const double fallback[3] = {
          center[0]-point[0], center[1]-point[1], center[2]-point[2]};
      const double norm = std::sqrt(fallback[0]*fallback[0] +
          fallback[1]*fallback[1] + fallback[2]*fallback[2]);
      if (norm <= 1.0e-12) {
        normal[0]=1.0; normal[1]=0.0; normal[2]=0.0;
      } else {
        normal[0]=fallback[0]/norm; normal[1]=fallback[1]/norm;
        normal[2]=fallback[2]/norm;
      }
      surface[0]=point[0]; surface[1]=point[1]; surface[2]=point[2];
      output_multipliers[item]=0.0;
      output_newton_iterations[item]=0;
      output_bisection_iterations[item]=0;
      output_residuals[item]=std::max(0.0, 1.0-inside);
      continue;
    }
    auto equation = [&](double value) {
      double result=-1.0;
      for (int axis=0; axis<3; ++axis) {
        const double denominator=value+values[axis];
        result += values[axis]*local[axis]*local[axis]/
            (denominator*denominator);
      }
      return result;
    };
    double lower=0.0;
    const double local_norm=std::sqrt(local[0]*local[0] +
        local[1]*local[1] + local[2]*local[2]);
    double upper=std::max(local_norm*std::sqrt(
        std::max(values[0], std::max(values[1], values[2]))), 1.0e-12);
    for (int guard=0; guard<128 && equation(upper)>0.0; ++guard)
      upper*=2.0;
    double multiplier=0.5*(lower+upper);
    if (initial_multipliers) {
      const double initial=initial_multipliers[item];
      if (std::isfinite(initial) && initial>lower && initial<upper)
        multiplier=initial;
    }
    int32_t newton_count=0;
    int32_t bisection_count=0;
    double residual=equation(multiplier);
    for (; newton_count<max_newton_iterations; ++newton_count) {
      residual=equation(multiplier);
      if (std::abs(residual)<=tolerance) {
        ++newton_count;
        break;
      }
      if (residual>0.0) lower=multiplier; else upper=multiplier;
      double derivative=0.0;
      for (int axis=0; axis<3; ++axis) {
        const double denominator=multiplier+values[axis];
        derivative += values[axis]*local[axis]*local[axis]/
            (denominator*denominator*denominator);
      }
      derivative*=-2.0;
      double candidate=multiplier-residual/derivative;
      if (!std::isfinite(candidate) || candidate<=lower || candidate>=upper) {
        candidate=0.5*(lower+upper);
        ++bisection_count;
      }
      multiplier=candidate;
    }
    residual=equation(multiplier);
    for (int32_t iteration=0;
         iteration<max_bisection_iterations && std::abs(residual)>tolerance;
         ++iteration) {
      if (residual>0.0) lower=multiplier; else upper=multiplier;
      multiplier=0.5*(lower+upper);
      ++bisection_count;
      residual=equation(multiplier);
    }
    double surface_local[3]{};
    for (int axis=0; axis<3; ++axis)
      surface_local[axis]=values[axis]*local[axis]/(multiplier+values[axis]);
    for (int row=0; row<3; ++row) {
      surface[row]=center[row] + rotation[3*row]*surface_local[0] +
          rotation[3*row+1]*surface_local[1] +
          rotation[3*row+2]*surface_local[2];
    }
    const double normal_delta[3] = {
        surface[0]-point[0], surface[1]-point[1], surface[2]-point[2]};
    const double normal_norm=std::sqrt(normal_delta[0]*normal_delta[0] +
        normal_delta[1]*normal_delta[1] + normal_delta[2]*normal_delta[2]);
    if (normal_norm<=1.0e-12) {
      const double fallback[3] = {
          center[0]-point[0], center[1]-point[1], center[2]-point[2]};
      const double norm=std::sqrt(fallback[0]*fallback[0] +
          fallback[1]*fallback[1] + fallback[2]*fallback[2]);
      if (norm<=1.0e-12) {
        normal[0]=1.0; normal[1]=0.0; normal[2]=0.0;
      } else {
        normal[0]=fallback[0]/norm; normal[1]=fallback[1]/norm;
        normal[2]=fallback[2]/norm;
      }
    } else {
      normal[0]=normal_delta[0]/normal_norm;
      normal[1]=normal_delta[1]/normal_norm;
      normal[2]=normal_delta[2]/normal_norm;
    }
    output_multipliers[item]=multiplier;
    output_newton_iterations[item]=newton_count;
    output_bisection_iterations[item]=bisection_count;
    output_residuals[item]=std::abs(residual);
  }
  return count;
}

// Same exact KKT solve as ellipsoid_closest_points_warm, but each row owns a
// distinct robot-sphere center.  A whole control tick can therefore cross the
// Python/ctypes boundary once and OpenMP can schedule thousands of independent
// pairs instead of repeatedly launching a small batch for every robot sphere.
MVT_API int32_t ellipsoid_closest_points_pairs_warm(
    const double* points, const double* obstacle_centers,
    const double* eigenvalues, const double* rotations,
    const double* initial_multipliers, int32_t count,
    int32_t max_newton_iterations, int32_t max_bisection_iterations,
    double tolerance, double* output_normals, double* output_surfaces,
    double* output_multipliers, int32_t* output_newton_iterations,
    int32_t* output_bisection_iterations, double* output_residuals) {
  if (!points || !obstacle_centers || !eigenvalues || !rotations ||
      !output_normals || !output_surfaces || !output_multipliers ||
      !output_newton_iterations || !output_bisection_iterations ||
      !output_residuals || count < 0 || max_newton_iterations < 0 ||
      max_bisection_iterations < 0 || !(tolerance > 0.0)) return -1;
  int32_t error_code = 0;
#pragma omp parallel for schedule(static) num_threads(4) if(count >= 64)
  for (int32_t item = 0; item < count; ++item) {
    const int32_t written = ellipsoid_closest_points_warm(
        points + 3*item, obstacle_centers + 3*item,
        eigenvalues + 3*item, rotations + 9*item,
        initial_multipliers ? initial_multipliers + item : nullptr, 1,
        max_newton_iterations, max_bisection_iterations, tolerance,
        output_normals + 3*item, output_surfaces + 3*item,
        output_multipliers + item, output_newton_iterations + item,
        output_bisection_iterations + item, output_residuals + item);
    if (written != 1) {
#pragma omp critical(ellipsoid_closest_pair_error)
      {
        if (error_code == 0) error_code = written;
      }
    }
  }
  return error_code == 0 ? count : error_code;
}

MVT_API int32_t ellipsoid_prune_planes(
    const double* obstacle_centers, const double* obstacle_shapes,
    const double* obstacle_offsets, const double* normals,
    const double* plane_offsets, int32_t count,
    int32_t* output_active, int32_t capacity) {
  if (!obstacle_centers || !obstacle_shapes || !obstacle_offsets || !normals ||
      !plane_offsets || !output_active || count < 0 || capacity < count) return -1;
  std::vector<int32_t> remaining(static_cast<size_t>(count));
  for (int32_t i=0; i<count; ++i) remaining[i]=i;
  int32_t written=0;
  constexpr double tolerance=1.0e-10;
  while (!remaining.empty()) {
    const int32_t nearest=remaining.front();
    output_active[written++]=nearest;
    const double* normal=normals+3*nearest;
    const double bound=plane_offsets[nearest]-tolerance;
    std::vector<int32_t> kept;
    kept.reserve(remaining.size());
    for (int32_t local : remaining) {
      if (local == nearest) continue;
      const double* center=obstacle_centers+3*local;
      const double* shape=obstacle_shapes+9*local;
      const double extent=support3(shape,normal)+obstacle_offsets[local];
      const double projection=center[0]*normal[0]+center[1]*normal[1]+center[2]*normal[2]-extent;
      if (projection < bound) kept.push_back(local);
    }
    remaining.swap(kept);
  }
  return written;
}

// Apply the identical ordered IRIS-inspired deletion independently to every
// robot sphere. group_offsets partitions one concatenated candidate array.
MVT_API int32_t ellipsoid_prune_planes_grouped(
    const double* obstacle_centers, const double* obstacle_shapes,
    const double* obstacle_offsets, const double* normals,
    const double* plane_offsets, const int32_t* group_offsets,
    int32_t group_count, int32_t total_count, int32_t* output_active,
    int32_t* output_counts) {
  if (!obstacle_centers || !obstacle_shapes || !obstacle_offsets || !normals ||
      !plane_offsets || !group_offsets || !output_active || !output_counts ||
      group_count < 0 || total_count < 0 || group_offsets[0] != 0 ||
      group_offsets[group_count] != total_count) return -1;
  int32_t error_code = 0;
#pragma omp parallel for schedule(static) num_threads(4) if(group_count >= 8)
  for (int32_t group = 0; group < group_count; ++group) {
    const int32_t begin = group_offsets[group];
    const int32_t end = group_offsets[group + 1];
    const int32_t count = end - begin;
    if (count < 0) {
#pragma omp critical(ellipsoid_prune_group_error)
      {
        if (error_code == 0) error_code = -2;
      }
      continue;
    }
    if (count == 0) {
      output_counts[group] = 0;
      continue;
    }
    const int32_t written = ellipsoid_prune_planes(
        obstacle_centers + 3*begin, obstacle_shapes + 9*begin,
        obstacle_offsets + begin, normals + 3*begin,
        plane_offsets + begin, count, output_active + begin, count);
    if (written < 0) {
#pragma omp critical(ellipsoid_prune_group_error)
      {
        if (error_code == 0) error_code = written;
      }
    } else {
      output_counts[group] = written;
    }
  }
  return error_code == 0 ? group_count : error_code;
}

// Exact ordered erase-remove without first solving every broad-phase pair.
// A conservative point-to-OBB lower bound identifies every pair that could be
// in CONTACT/RECOVERY; those pairs are solved up front and re-added if an
// earlier separating plane removes them. All remaining closest points are
// solved lazily only when they become the first surviving LiuQP candidate.
MVT_API int32_t ellipsoid_closest_prune_pairs_warm(
    const double* robot_centers, const double* robot_radii,
    int32_t* obstacle_indices, int32_t obstacle_count,
    const double* obstacle_centers, const double* eigenvalues,
    const double* rotations,
    const double* obstacle_shapes, const double* obstacle_offsets,
    double* multiplier_dense,
    const int32_t* group_offsets, int32_t group_count, int32_t total_count,
    int32_t max_newton_iterations, int32_t max_bisection_iterations,
    double root_tolerance, double contact_distance,
    double* output_normals, double* output_surfaces,
    double* output_multipliers, int32_t* output_newton_iterations,
    int32_t* output_bisection_iterations, double* output_residuals,
    int32_t* output_computed, int32_t* output_warm_start,
    int32_t* output_active,
    int32_t* output_counts) {
  if (!robot_centers || !robot_radii || !obstacle_indices ||
      obstacle_count < 0 || !obstacle_centers || !eigenvalues || !rotations ||
      !obstacle_shapes || !obstacle_offsets || !multiplier_dense ||
      !group_offsets || !output_normals || !output_surfaces ||
      !output_multipliers || !output_newton_iterations ||
      !output_bisection_iterations || !output_residuals ||
      !output_computed || !output_warm_start || !output_active ||
      !output_counts ||
      group_count < 0 || total_count < 0 || group_offsets[0] != 0 ||
      group_offsets[group_count] != total_count ||
      max_newton_iterations < 0 || max_bisection_iterations < 0 ||
      !(root_tolerance > 0.0)) return -1;
  int32_t error_code = 0;
  constexpr double plane_tolerance = 1.0e-10;
#pragma omp parallel for schedule(static) num_threads(g_ellipsoid_pair_threads) if(group_count >= 8)
  for (int32_t group = 0; group < group_count; ++group) {
#ifdef _WIN32
    if (g_ellipsoid_pair_affinity_mask != 0) {
      SetThreadAffinityMask(
          GetCurrentThread(),
          static_cast<DWORD_PTR>(g_ellipsoid_pair_affinity_mask));
    }
#endif
    const int32_t begin = group_offsets[group];
    const int32_t end = group_offsets[group + 1];
    const int32_t count = end - begin;
    if (count < 0) {
#pragma omp critical(ellipsoid_lazy_pair_error)
      {
        if (error_code == 0) error_code = -2;
      }
      continue;
    }
    const double* robot_center = robot_centers + 3*group;
    // Compute the frozen ordering key once per pair.  The previous comparator
    // recomputed two square roots for every O(N log N) comparison; caching the
    // identical key changes neither candidate membership nor ordering.
    struct OrderedObstacle {
      int32_t obstacle;
      double lower_bound;
    };
    std::vector<OrderedObstacle> ordered(static_cast<size_t>(count));
    for (int32_t local=0; local<count; ++local) {
      const int32_t obstacle=obstacle_indices[begin+local];
      if (obstacle < 0 || obstacle >= obstacle_count) {
#pragma omp critical(ellipsoid_lazy_pair_error)
        {
          if (error_code == 0) error_code = -5;
        }
        ordered[static_cast<size_t>(local)] = {obstacle, 0.0};
        continue;
      }
      const double* center=obstacle_centers+3*obstacle;
      const double dx=center[0]-robot_center[0];
      const double dy=center[1]-robot_center[1];
      const double dz=center[2]-robot_center[2];
      const double* values=eigenvalues+3*obstacle;
      const double extent=std::sqrt(std::max(
          values[0], std::max(values[1], values[2]))) +
          obstacle_offsets[obstacle];
      ordered[static_cast<size_t>(local)] = {
          obstacle, std::sqrt(dx*dx+dy*dy+dz*dz)-extent};
    }
    std::sort(ordered.begin(), ordered.end(),
        [](const OrderedObstacle& first, const OrderedObstacle& second) {
          return first.lower_bound < second.lower_bound ||
              (first.lower_bound == second.lower_bound &&
               first.obstacle < second.obstacle);
        });
    for (int32_t local=0; local<count; ++local)
      obstacle_indices[begin+local]=ordered[static_cast<size_t>(local)].obstacle;
    std::vector<uint8_t> computed(static_cast<size_t>(count), 0);
    std::vector<uint8_t> contact(static_cast<size_t>(count), 0);
    std::vector<uint8_t> active_mask(static_cast<size_t>(count), 0);
    for (int32_t local = 0; local < count; ++local) {
      output_computed[begin + local] = 0;
      output_warm_start[begin + local] = 0;
    }

    auto solve_local = [&](int32_t local) -> bool {
      if (computed[static_cast<size_t>(local)]) return true;
      const int32_t item = begin + local;
      const int32_t obstacle = obstacle_indices[item];
      if (obstacle < 0 || obstacle >= obstacle_count) return false;
      const size_t dense_index =
          static_cast<size_t>(group)*obstacle_count + obstacle;
      output_warm_start[item] =
          std::isfinite(multiplier_dense[dense_index]) ? 1 : 0;
      const int32_t written = ellipsoid_closest_points_warm(
          robot_centers + 3*group, obstacle_centers + 3*obstacle,
          eigenvalues + 3*obstacle, rotations + 9*obstacle,
          multiplier_dense + dense_index,
          1,
          max_newton_iterations, max_bisection_iterations, root_tolerance,
          output_normals + 3*item, output_surfaces + 3*item,
          output_multipliers + item, output_newton_iterations + item,
          output_bisection_iterations + item, output_residuals + item);
      if (written != 1) return false;
      computed[static_cast<size_t>(local)] = 1;
      output_computed[item] = 1;
      multiplier_dense[dense_index] = output_multipliers[item];
      return true;
    };

    // OBB contains the ellipsoid, hence its distance is a lower bound on the
    // true point-to-ellipsoid distance. No possible contact can be missed.
    for (int32_t local = 0; local < count; ++local) {
      const int32_t item = begin + local;
      const int32_t obstacle = obstacle_indices[item];
      if (obstacle < 0 || obstacle >= obstacle_count) {
#pragma omp critical(ellipsoid_lazy_pair_error)
        {
          if (error_code == 0) error_code = -5;
        }
        continue;
      }
      const double* point = robot_centers + 3*group;
      const double* center = obstacle_centers + 3*obstacle;
      const double* values = eigenvalues + 3*obstacle;
      const double* rotation = rotations + 9*obstacle;
      const double delta[3] = {
          point[0]-center[0], point[1]-center[1], point[2]-center[2]};
      double outside_squared = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double coordinate =
            rotation[axis]*delta[0] + rotation[3+axis]*delta[1] +
            rotation[6+axis]*delta[2];
        const double excess = std::max(
            std::abs(coordinate)-std::sqrt(std::max(values[axis], 0.0)), 0.0);
        outside_squared += excess*excess;
      }
      const double lower_surface_clearance =
          std::sqrt(outside_squared) - obstacle_offsets[obstacle] -
          robot_radii[group];
      if (lower_surface_clearance <= contact_distance) {
        if (!solve_local(local)) {
#pragma omp critical(ellipsoid_lazy_pair_error)
          {
            if (error_code == 0) error_code = -3;
          }
          continue;
        }
        const double* surface = output_surfaces + 3*item;
        const double dx=surface[0]-point[0];
        const double dy=surface[1]-point[1];
        const double dz=surface[2]-point[2];
        const double clearance=std::sqrt(dx*dx+dy*dy+dz*dz) -
            obstacle_offsets[obstacle] - robot_radii[group];
        if (clearance <= contact_distance)
          contact[static_cast<size_t>(local)] = 1;
      }
    }

    std::vector<int32_t> remaining(static_cast<size_t>(count));
    for (int32_t local=0; local<count; ++local) remaining[local]=local;
    int32_t written=0;
    while (!remaining.empty()) {
      const int32_t nearest=remaining.front();
      if (!solve_local(nearest)) {
#pragma omp critical(ellipsoid_lazy_pair_error)
        {
          if (error_code == 0) error_code = -4;
        }
        break;
      }
      const int32_t item=begin+nearest;
      const int32_t obstacle=obstacle_indices[item];
      output_active[begin+written++]=nearest;
      active_mask[static_cast<size_t>(nearest)]=1;
      const double* normal=output_normals+3*item;
      const double* surface=output_surfaces+3*item;
      const double plane_offset =
          normal[0]*(surface[0]-obstacle_offsets[obstacle]*normal[0]) +
          normal[1]*(surface[1]-obstacle_offsets[obstacle]*normal[1]) +
          normal[2]*(surface[2]-obstacle_offsets[obstacle]*normal[2]);
      const double bound=plane_offset-plane_tolerance;
      std::vector<int32_t> kept;
      kept.reserve(remaining.size());
      for (int32_t candidate : remaining) {
        if (candidate == nearest) continue;
        const int32_t candidate_item=begin+candidate;
        const int32_t candidate_obstacle=obstacle_indices[candidate_item];
        const double* center=obstacle_centers+3*candidate_obstacle;
        const double* shape=obstacle_shapes+9*candidate_obstacle;
        const double extent=support3(shape,normal)+
            obstacle_offsets[candidate_obstacle];
        const double projection=center[0]*normal[0]+center[1]*normal[1]+
            center[2]*normal[2]-extent;
        if (projection < bound) kept.push_back(candidate);
      }
      remaining.swap(kept);
    }
    // Match the Python state-machine rule: CONTACT/RECOVERY has priority over
    // geometric redundancy and is appended in original candidate order.
    for (int32_t local=0; local<count; ++local) {
      if (contact[static_cast<size_t>(local)] &&
          !active_mask[static_cast<size_t>(local)])
        output_active[begin+written++]=local;
    }
    output_counts[group]=written;
  }
  return error_code == 0 ? group_count : error_code;
}

// Fused real-time path: keep the 27-cell MVT/SIMD candidates inside native
// memory and feed them directly to the identical exact ordered erase-remove
// kernel above.  Query membership, closest points, state priority and active
// rows are unchanged; only Python round-trips and intermediate sorting vanish.
MVT_API int32_t mvt_ellipsoid_closest_prune_pairs_warm(
    void* mvt_handle,
    const double* robot_centers, const double* robot_radii,
    double query_padding, int32_t group_count,
    int32_t candidate_capacity,
    int32_t* obstacle_indices, int32_t* group_offsets,
    int32_t obstacle_count,
    const double* obstacle_centers, const double* eigenvalues,
    const double* rotations,
    const double* obstacle_shapes, const double* obstacle_offsets,
    double* multiplier_dense,
    int32_t max_newton_iterations, int32_t max_bisection_iterations,
    double root_tolerance, double contact_distance,
    double* output_normals, double* output_surfaces,
    double* output_multipliers, int32_t* output_newton_iterations,
    int32_t* output_bisection_iterations, double* output_residuals,
    int32_t* output_computed, int32_t* output_warm_start,
    int32_t* output_active, int32_t* output_counts) {
  auto* mvt=static_cast<NativeMultilevelMVT*>(mvt_handle);
  if (!mvt || !robot_centers || !robot_radii || !(query_padding >= 0.0) ||
      group_count < 0 || candidate_capacity <= 0 || !obstacle_indices ||
      !group_offsets || obstacle_count != mvt->n) return -10;
  int32_t total_count=0;
  group_offsets[0]=0;
  for (int32_t group=0; group<group_count; ++group) {
    const float center[3] = {
        static_cast<float>(robot_centers[3*group]),
        static_cast<float>(robot_centers[3*group+1]),
        static_cast<float>(robot_centers[3*group+2])};
    const float radius=static_cast<float>(robot_radii[group]+query_padding);
    const int32_t count=mvt_multilevel_query_sphere_impl(
        mvt, center, radius, obstacle_indices+total_count,
        candidate_capacity-total_count, true, false);
    if (count == -2) return -20;
    if (count < 0) return -21;
    total_count+=count;
    group_offsets[group+1]=total_count;
  }
  const int32_t result=ellipsoid_closest_prune_pairs_warm(
      robot_centers, robot_radii, obstacle_indices, obstacle_count,
      obstacle_centers, eigenvalues, rotations, obstacle_shapes,
      obstacle_offsets, multiplier_dense, group_offsets, group_count,
      total_count, max_newton_iterations, max_bisection_iterations,
      root_tolerance, contact_distance, output_normals, output_surfaces,
      output_multipliers, output_newton_iterations,
      output_bisection_iterations, output_residuals, output_computed,
      output_warm_start, output_active, output_counts);
  if (result != group_count) return result - 100;
  return total_count;
}

// Exact directional-uncertainty real-time path for
//   B(robot_radius) (+) E(Q) (+) E(U) (+) B(offset).
// MVT supplies the same AVX2 ball--AABB candidate set as the eager reference.
// Candidates retain the frozen lower-bound ordering, but safeguarded Newton is
// evaluated lazily: only the first still-surviving candidate and every pair
// that can conservatively be in CONTACT/RECOVERY are solved.  Plane deletion
// uses the exact support sum, so this changes computation order only, not the
// final LiuQP rows.
MVT_API int32_t mvt_ellipsoid_support_sum_prune_pairs_warm(
    void* mvt_handle,
    const double* robot_centers, const double* robot_radii,
    double query_padding, int32_t group_count,
    int32_t candidate_capacity,
    int32_t* obstacle_indices, int32_t* group_offsets,
    int32_t obstacle_count,
    const double* obstacle_centers, const double* obstacle_shapes,
    const double* uncertainty_shapes,
    const double* obstacle_eigenvalues,
    const double* uncertainty_eigenvalues,
    const double* obstacle_offsets, const int64_t* proxy_ids,
    int32_t normal_capacity, double* normal_dense,
    int32_t max_newton_iterations, int32_t retry_newton_iterations,
    double contact_distance,
    double* output_normals, double* output_surfaces,
    int32_t* output_iterations, double* output_residuals,
    int32_t* output_computed, int32_t* output_warm_start,
    int32_t* output_active, int32_t* output_counts) {
  auto* mvt=static_cast<NativeMultilevelMVT*>(mvt_handle);
  if (!mvt || !robot_centers || !robot_radii || !(query_padding >= 0.0) ||
      group_count < 0 || candidate_capacity <= 0 || !obstacle_indices ||
      !group_offsets || obstacle_count != mvt->n || !obstacle_centers ||
      !obstacle_shapes || !uncertainty_shapes || !obstacle_eigenvalues ||
      !uncertainty_eigenvalues || !obstacle_offsets || !normal_dense ||
      !proxy_ids || normal_capacity <= 0 ||
      max_newton_iterations < 0 || retry_newton_iterations < 0 ||
      !output_normals || !output_surfaces || !output_iterations ||
      !output_residuals || !output_computed || !output_warm_start ||
      !output_active || !output_counts) return -10;

  int32_t total_count=0;
  group_offsets[0]=0;
  for (int32_t group=0; group<group_count; ++group) {
    const float center[3] = {
        static_cast<float>(robot_centers[3*group]),
        static_cast<float>(robot_centers[3*group+1]),
        static_cast<float>(robot_centers[3*group+2])};
    const float radius=static_cast<float>(robot_radii[group]+query_padding);
    const int32_t count=mvt_multilevel_query_sphere_impl(
        mvt, center, radius, obstacle_indices+total_count,
        candidate_capacity-total_count, true, false);
    if (count == -2) return -20;
    if (count < 0) return -21;
    total_count+=count;
    group_offsets[group+1]=total_count;
  }

  int32_t error_code=0;
  constexpr double plane_tolerance=1.0e-10;
  constexpr double residual_gate=1.0e-7;
#pragma omp parallel for schedule(static) num_threads(g_ellipsoid_pair_threads) if(group_count >= 8)
  for (int32_t group=0; group<group_count; ++group) {
#ifdef _WIN32
    if (g_ellipsoid_pair_affinity_mask != 0) {
      SetThreadAffinityMask(
          GetCurrentThread(),
          static_cast<DWORD_PTR>(g_ellipsoid_pair_affinity_mask));
    }
#endif
    const int32_t begin=group_offsets[group];
    const int32_t end=group_offsets[group+1];
    const int32_t count=end-begin;
    if (count < 0) {
#pragma omp critical(ellipsoid_support_sum_lazy_error)
      {
        if (error_code == 0) error_code=-22;
      }
      continue;
    }
    const double* robot_center=robot_centers+3*group;
    const double robot_radius=robot_radii[group];
    const double radius_squared=robot_radius*robot_radius;
    const double robot_shape[9] = {
        radius_squared,0.0,0.0,0.0,radius_squared,0.0,
        0.0,0.0,radius_squared};
    struct OrderedObstacle {
      int32_t obstacle;
      double lower_bound;
    };
    std::vector<OrderedObstacle> ordered(static_cast<size_t>(count));
    for (int32_t local=0; local<count; ++local) {
      const int32_t obstacle=obstacle_indices[begin+local];
      if (obstacle < 0 || obstacle >= obstacle_count) {
#pragma omp critical(ellipsoid_support_sum_lazy_error)
        {
          if (error_code == 0) error_code=-23;
        }
        ordered[static_cast<size_t>(local)]={obstacle,0.0};
        continue;
      }
      const double* center=obstacle_centers+3*obstacle;
      const double dx=center[0]-robot_center[0];
      const double dy=center[1]-robot_center[1];
      const double dz=center[2]-robot_center[2];
      const double* values=obstacle_eigenvalues+3*obstacle;
      const double* uncertainty_values=uncertainty_eigenvalues+3*obstacle;
      const double obstacle_extent=std::sqrt(std::max(
          0.0,std::max(values[0],std::max(values[1],values[2]))));
      const double uncertainty_extent=std::sqrt(std::max(
          0.0,std::max(uncertainty_values[0],
                       std::max(uncertainty_values[1],
                                uncertainty_values[2]))));
      const double extent=obstacle_extent+uncertainty_extent+
          obstacle_offsets[obstacle];
      ordered[static_cast<size_t>(local)]={
          obstacle,std::sqrt(dx*dx+dy*dy+dz*dz)-extent-robot_radius};
    }
    std::sort(
        ordered.begin(),ordered.end(),
        [](const OrderedObstacle& first,const OrderedObstacle& second) {
          return first.lower_bound < second.lower_bound ||
              (first.lower_bound == second.lower_bound &&
               first.obstacle < second.obstacle);
        });
    for (int32_t local=0; local<count; ++local)
      obstacle_indices[begin+local]=ordered[static_cast<size_t>(local)].obstacle;

    std::vector<uint8_t> computed(static_cast<size_t>(count),0);
    std::vector<uint8_t> contact(static_cast<size_t>(count),0);
    std::vector<uint8_t> active_mask(static_cast<size_t>(count),0);
    for (int32_t local=0; local<count; ++local) {
      output_computed[begin+local]=0;
      output_warm_start[begin+local]=0;
    }

    auto solve_local = [&](int32_t local) -> bool {
      if (computed[static_cast<size_t>(local)]) return true;
      const int32_t item=begin+local;
      const int32_t obstacle=obstacle_indices[item];
      if (obstacle < 0 || obstacle >= obstacle_count) return false;
      const int64_t proxy_id=proxy_ids[obstacle];
      if (proxy_id < 0 || proxy_id >= normal_capacity) return false;
      const size_t dense_index=(
          static_cast<size_t>(group)*normal_capacity+
          static_cast<size_t>(proxy_id))*3;
      double* cached=normal_dense+dense_index;
      const double cached_norm=std::sqrt(
          cached[0]*cached[0]+cached[1]*cached[1]+cached[2]*cached[2]);
      const bool warm=std::isfinite(cached_norm) && cached_norm>1.0e-12;
      output_warm_start[item]=warm ? 1 : 0;
      ellipsoid_support_normal_sum_newton_one(
          robot_center,robot_shape,obstacle_centers+3*obstacle,
          obstacle_shapes+9*obstacle,uncertainty_shapes+9*obstacle,
          warm ? cached : nullptr,max_newton_iterations,
          output_normals+3*item,output_iterations+item,
          output_residuals+item);
      if (output_residuals[item] > residual_gate) {
        int32_t retry_iterations=0;
        double retry_residual=0.0;
        double retry_normal[3]{};
        ellipsoid_support_normal_sum_newton_one(
            robot_center,robot_shape,obstacle_centers+3*obstacle,
            obstacle_shapes+9*obstacle,uncertainty_shapes+9*obstacle,
            nullptr,retry_newton_iterations,retry_normal,&retry_iterations,
            &retry_residual);
        output_normals[3*item]=retry_normal[0];
        output_normals[3*item+1]=retry_normal[1];
        output_normals[3*item+2]=retry_normal[2];
        output_iterations[item]+=retry_iterations;
        output_residuals[item]=retry_residual;
      }
      if (output_residuals[item] > residual_gate) return false;
      const double* normal=output_normals+3*item;
      const double* shape=obstacle_shapes+9*obstacle;
      const double* uncertainty=uncertainty_shapes+9*obstacle;
      const double obstacle_extent=std::max(support3(shape,normal),1.0e-12);
      const double uncertainty_extent=std::max(
          support3(uncertainty,normal),1.0e-12);
      double* surface=output_surfaces+3*item;
      for (int axis=0; axis<3; ++axis) {
        const double qn=shape[3*axis]*normal[0]+
            shape[3*axis+1]*normal[1]+shape[3*axis+2]*normal[2];
        const double un=uncertainty[3*axis]*normal[0]+
            uncertainty[3*axis+1]*normal[1]+
            uncertainty[3*axis+2]*normal[2];
        surface[axis]=obstacle_centers[3*obstacle+axis]-
            qn/obstacle_extent-un/uncertainty_extent-
            obstacle_offsets[obstacle]*normal[axis];
      }
      cached[0]=normal[0]; cached[1]=normal[1]; cached[2]=normal[2];
      computed[static_cast<size_t>(local)]=1;
      output_computed[item]=1;
      return true;
    };

    // The enclosing sphere of E(Q) (+) E(U) (+) B(offset) gives a
    // conservative lower clearance.  Every possible contact is solved and
    // later restored even if an earlier plane geometrically removes it.
    for (int32_t local=0; local<count; ++local) {
      const int32_t item=begin+local;
      const int32_t obstacle=obstacle_indices[item];
      if (obstacle < 0 || obstacle >= obstacle_count) continue;
      const double* center=obstacle_centers+3*obstacle;
      const double dx=center[0]-robot_center[0];
      const double dy=center[1]-robot_center[1];
      const double dz=center[2]-robot_center[2];
      const double center_distance=std::sqrt(dx*dx+dy*dy+dz*dz);
      double centerline_gap=-std::numeric_limits<double>::infinity();
      if (center_distance > 1.0e-15) {
        const double direction[3] = {
            dx/center_distance,dy/center_distance,dz/center_distance};
        // For centrally symmetric convex sets, the exact separation is the
        // maximum support-plane gap over unit normals.  The center-line gap is
        // therefore a certified lower bound: when it already exceeds the
        // CONTACT threshold, exact Newton cannot classify this pair as
        // contact.  This is strictly tighter than the old enclosing-sphere
        // test while remaining conservative.
        centerline_gap=center_distance-robot_radius-
            support3(obstacle_shapes+9*obstacle,direction)-
            support3(uncertainty_shapes+9*obstacle,direction)-
            obstacle_offsets[obstacle];
      }
      if (centerline_gap <= contact_distance) {
        if (!solve_local(local)) {
#pragma omp critical(ellipsoid_support_sum_lazy_error)
          {
            if (error_code == 0) error_code=-24;
          }
          continue;
        }
        const double* normal=output_normals+3*item;
        const double clearance=normal[0]*(obstacle_centers[3*obstacle]-robot_center[0])+
            normal[1]*(obstacle_centers[3*obstacle+1]-robot_center[1])+
            normal[2]*(obstacle_centers[3*obstacle+2]-robot_center[2])-
            robot_radius-support3(obstacle_shapes+9*obstacle,normal)-
            support3(uncertainty_shapes+9*obstacle,normal)-
            obstacle_offsets[obstacle];
        if (clearance <= contact_distance)
          contact[static_cast<size_t>(local)]=1;
      }
    }

    std::vector<int32_t> remaining(static_cast<size_t>(count));
    for (int32_t local=0; local<count; ++local) remaining[local]=local;
    int32_t written=0;
    while (!remaining.empty()) {
      const int32_t nearest=remaining.front();
      if (!solve_local(nearest)) {
#pragma omp critical(ellipsoid_support_sum_lazy_error)
        {
          if (error_code == 0) error_code=-25;
        }
        break;
      }
      const int32_t item=begin+nearest;
      output_active[begin+written++]=nearest;
      active_mask[static_cast<size_t>(nearest)]=1;
      const double* normal=output_normals+3*item;
      const double* surface=output_surfaces+3*item;
      const double plane_offset=normal[0]*surface[0]+normal[1]*surface[1]+
          normal[2]*surface[2];
      const double bound=plane_offset-plane_tolerance;
      std::vector<int32_t> kept;
      kept.reserve(remaining.size());
      for (int32_t candidate : remaining) {
        if (candidate == nearest) continue;
        const int32_t candidate_obstacle=obstacle_indices[begin+candidate];
        const double* candidate_center=obstacle_centers+3*candidate_obstacle;
        const double projection=candidate_center[0]*normal[0]+
            candidate_center[1]*normal[1]+candidate_center[2]*normal[2]-
            support3(obstacle_shapes+9*candidate_obstacle,normal)-
            support3(uncertainty_shapes+9*candidate_obstacle,normal)-
            obstacle_offsets[candidate_obstacle];
        if (projection < bound) kept.push_back(candidate);
      }
      remaining.swap(kept);
    }
    for (int32_t local=0; local<count; ++local) {
      if (contact[static_cast<size_t>(local)] &&
          !active_mask[static_cast<size_t>(local)])
        output_active[begin+written++]=local;
    }
    output_counts[group]=written;
  }
  if (error_code != 0) return error_code;
  return total_count;
}

// Matched-sphere real-time path.  Candidate membership is the same
// multilevel 27-cell AVX2 ball--AABB query used above.  The narrow phase uses
// the analytic sphere closest point and the exact LiuQP ordered erase-remove
// rule; only the implementation language and batching differ from the Python
// reference.
MVT_API int32_t mvt_sphere_prune_pairs(
    void* mvt_handle,
    const double* robot_centers, const double* robot_radii,
    double query_padding, int32_t group_count,
    int32_t candidate_capacity,
    int32_t* obstacle_indices, int32_t* group_offsets,
    int32_t obstacle_count,
    const double* obstacle_centers, const double* obstacle_radii,
    const double* obstacle_offsets, double contact_distance,
    double* output_normals, double* output_surfaces,
    int32_t* output_active, int32_t* output_counts) {
  auto* mvt=static_cast<NativeMultilevelMVT*>(mvt_handle);
  if (!mvt || !robot_centers || !robot_radii || !(query_padding >= 0.0) ||
      group_count < 0 || candidate_capacity <= 0 || !obstacle_indices ||
      !group_offsets || obstacle_count != mvt->n || !obstacle_centers ||
      !obstacle_radii || !obstacle_offsets || !output_normals ||
      !output_surfaces || !output_active || !output_counts) return -10;

  int32_t total_count=0;
  group_offsets[0]=0;
  for (int32_t group=0; group<group_count; ++group) {
    const float center[3] = {
        static_cast<float>(robot_centers[3*group]),
        static_cast<float>(robot_centers[3*group+1]),
        static_cast<float>(robot_centers[3*group+2])};
    const float radius=static_cast<float>(robot_radii[group]+query_padding);
    const int32_t count=mvt_multilevel_query_sphere_impl(
        mvt, center, radius, obstacle_indices+total_count,
        candidate_capacity-total_count, true, false);
    if (count == -2) return -20;
    if (count < 0) return -21;
    total_count+=count;
    group_offsets[group+1]=total_count;
  }

  int32_t error_code=0;
  constexpr double plane_tolerance=1.0e-10;
#pragma omp parallel for schedule(static) num_threads(g_ellipsoid_pair_threads) if(group_count >= 8)
  for (int32_t group=0; group<group_count; ++group) {
#ifdef _WIN32
    if (g_ellipsoid_pair_affinity_mask != 0) {
      SetThreadAffinityMask(
          GetCurrentThread(),
          static_cast<DWORD_PTR>(g_ellipsoid_pair_affinity_mask));
    }
#endif
    const int32_t begin=group_offsets[group];
    const int32_t end=group_offsets[group+1];
    const int32_t count=end-begin;
    if (count < 0) {
#pragma omp critical(sphere_pair_error)
      {
        if (error_code == 0) error_code=-22;
      }
      continue;
    }
    const double* robot_center=robot_centers+3*group;
    struct OrderedSphere {
      int32_t obstacle;
      double lower_bound;
    };
    std::vector<OrderedSphere> ordered(static_cast<size_t>(count));
    for (int32_t local=0; local<count; ++local) {
      const int32_t obstacle=obstacle_indices[begin+local];
      if (obstacle < 0 || obstacle >= obstacle_count) {
#pragma omp critical(sphere_pair_error)
        {
          if (error_code == 0) error_code=-23;
        }
        ordered[static_cast<size_t>(local)]={obstacle,0.0};
        continue;
      }
      const double* center=obstacle_centers+3*obstacle;
      const double dx=center[0]-robot_center[0];
      const double dy=center[1]-robot_center[1];
      const double dz=center[2]-robot_center[2];
      const double extent=obstacle_radii[obstacle]+obstacle_offsets[obstacle];
      ordered[static_cast<size_t>(local)]={
          obstacle,std::sqrt(dx*dx+dy*dy+dz*dz)-extent};
    }
    std::sort(
        ordered.begin(),ordered.end(),
        [](const OrderedSphere& first,const OrderedSphere& second) {
          return first.lower_bound < second.lower_bound ||
              (first.lower_bound == second.lower_bound &&
               first.obstacle < second.obstacle);
        });

    std::vector<uint8_t> contact(static_cast<size_t>(count),0);
    std::vector<uint8_t> active_mask(static_cast<size_t>(count),0);
    for (int32_t local=0; local<count; ++local) {
      const int32_t item=begin+local;
      const int32_t obstacle=ordered[static_cast<size_t>(local)].obstacle;
      obstacle_indices[item]=obstacle;
      if (obstacle < 0 || obstacle >= obstacle_count) continue;
      const double* center=obstacle_centers+3*obstacle;
      const double dx=center[0]-robot_center[0];
      const double dy=center[1]-robot_center[1];
      const double dz=center[2]-robot_center[2];
      const double distance=std::sqrt(dx*dx+dy*dy+dz*dz);
      double* normal=output_normals+3*item;
      if (distance <= 1.0e-12) {
        normal[0]=1.0;
        normal[1]=0.0;
        normal[2]=0.0;
      } else {
        normal[0]=dx/distance;
        normal[1]=dy/distance;
        normal[2]=dz/distance;
      }
      // Store the base sphere surface.  The Python row builder applies the
      // independent scalar offset exactly as in the reference path.
      double* surface=output_surfaces+3*item;
      surface[0]=center[0]-obstacle_radii[obstacle]*normal[0];
      surface[1]=center[1]-obstacle_radii[obstacle]*normal[1];
      surface[2]=center[2]-obstacle_radii[obstacle]*normal[2];
      const double clearance=distance-obstacle_radii[obstacle]-
          obstacle_offsets[obstacle]-robot_radii[group];
      if (clearance <= contact_distance)
        contact[static_cast<size_t>(local)]=1;
    }

    std::vector<int32_t> remaining(static_cast<size_t>(count));
    for (int32_t local=0; local<count; ++local) remaining[local]=local;
    int32_t written=0;
    while (!remaining.empty()) {
      const int32_t nearest=remaining.front();
      const int32_t item=begin+nearest;
      const int32_t obstacle=obstacle_indices[item];
      output_active[begin+written++]=nearest;
      active_mask[static_cast<size_t>(nearest)]=1;
      const double* normal=output_normals+3*item;
      const double* center=obstacle_centers+3*obstacle;
      const double effective_radius=
          obstacle_radii[obstacle]+obstacle_offsets[obstacle];
      const double plane_offset=center[0]*normal[0]+center[1]*normal[1]+
          center[2]*normal[2]-effective_radius;
      const double bound=plane_offset-plane_tolerance;
      std::vector<int32_t> kept;
      kept.reserve(remaining.size());
      for (int32_t candidate : remaining) {
        if (candidate == nearest) continue;
        const int32_t candidate_obstacle=
            obstacle_indices[begin+candidate];
        const double* candidate_center=
            obstacle_centers+3*candidate_obstacle;
        const double candidate_extent=
            obstacle_radii[candidate_obstacle]+
            obstacle_offsets[candidate_obstacle];
        const double projection=
            candidate_center[0]*normal[0]+
            candidate_center[1]*normal[1]+
            candidate_center[2]*normal[2]-candidate_extent;
        if (projection < bound) kept.push_back(candidate);
      }
      remaining.swap(kept);
    }
    // Preserve the LiuQP CONTACT/RECOVERY priority used by the formal
    // ellipsoid path.  In collision-free states this appends nothing.
    for (int32_t local=0; local<count; ++local) {
      if (contact[static_cast<size_t>(local)] &&
          !active_mask[static_cast<size_t>(local)])
        output_active[begin+written++]=local;
    }
    output_counts[group]=written;
  }
  if (error_code != 0) return error_code;
  return total_count;
}

// Directional-uncertainty extension of the exact same ordered LiuQP
// erase/remove rule above.  An obstacle certificate is the Minkowski sum
// E(Q) (+) E(U) (+) B(offset), so its minimum support in n is
// c.n - sqrt(n'Qn) - sqrt(n'Un) - offset.  The input order, tolerance, and
// first-remaining selection exactly match the Python reference implementation.
MVT_API int32_t ellipsoid_prune_planes_sum(
    const double* obstacle_centers, const double* obstacle_shapes,
    const double* uncertainty_shapes, const double* obstacle_offsets,
    const double* normals, const double* plane_offsets, int32_t count,
    int32_t* output_active, int32_t capacity) {
  if (!obstacle_centers || !obstacle_shapes || !uncertainty_shapes ||
      !obstacle_offsets || !normals || !plane_offsets || !output_active ||
      count < 0 || capacity < count) return -1;
  std::vector<int32_t> remaining(static_cast<size_t>(count));
  for (int32_t i=0; i<count; ++i) remaining[i]=i;
  int32_t written=0;
  constexpr double tolerance=1.0e-10;
  while (!remaining.empty()) {
    const int32_t nearest=remaining.front();
    output_active[written++]=nearest;
    const double* normal=normals+3*nearest;
    const double bound=plane_offsets[nearest]-tolerance;
    std::vector<int32_t> kept;
    kept.reserve(remaining.size());
    for (int32_t local : remaining) {
      if (local == nearest) continue;
      const double* center=obstacle_centers+3*local;
      const double* shape=obstacle_shapes+9*local;
      const double* uncertainty=uncertainty_shapes+9*local;
      const double extent=support3(shape,normal)+support3(uncertainty,normal)
          +obstacle_offsets[local];
      const double projection=center[0]*normal[0]+center[1]*normal[1]
          +center[2]*normal[2]-extent;
      if (projection < bound) kept.push_back(local);
    }
    remaining.swap(kept);
  }
  return written;
}

MVT_API void* occ_incremental_create(
    double voxel_size, int32_t free_decrement, int32_t occupied_increment,
    int32_t minimum_score, int32_t maximum_score,
    int32_t occupied_threshold, int32_t free_threshold) {
  if (!(voxel_size > 0.0) || free_decrement <= 0 ||
      occupied_increment <= 0 || minimum_score >= maximum_score) return nullptr;
  try {
    auto* map = new NativeIncrementalOccupancy();
    map->voxel = voxel_size;
    map->free_decrement = static_cast<int8_t>(free_decrement);
    map->occupied_increment = static_cast<int8_t>(occupied_increment);
    map->minimum_score = static_cast<int8_t>(minimum_score);
    map->maximum_score = static_cast<int8_t>(maximum_score);
    map->occupied_threshold = static_cast<int8_t>(occupied_threshold);
    map->free_threshold = static_cast<int8_t>(free_threshold);
    map->scores.reserve(131072);
    map->dirty_state_keys.reserve(131072);
    return map;
  } catch (...) {
    return nullptr;
  }
}

MVT_API void occ_incremental_destroy(void* handle) {
  delete static_cast<NativeIncrementalOccupancy*>(handle);
}

MVT_API int32_t occ_incremental_set_state_journal_enabled(
    void* handle, int32_t enabled) {
  auto* map = static_cast<NativeIncrementalOccupancy*>(handle);
  if (!map) return -1;
  map->state_journal_enabled = enabled != 0;
  if (!map->state_journal_enabled) map->dirty_state_keys.clear();
  return 0;
}

MVT_API int32_t occ_incremental_set_calibrated_free_aabb(
    void* handle, const double* lower, const double* upper) {
  auto* map = static_cast<NativeIncrementalOccupancy*>(handle);
  if (!map || !lower || !upper) return -1;
  for (int axis = 0; axis < 3; ++axis) {
    if (!(lower[axis] < upper[axis])) return -2;
    map->calibrated_free_lo[axis] = lower[axis];
    map->calibrated_free_hi[axis] = upper[axis];
  }
  map->has_calibrated_free_aabb = true;
  return 0;
}

template <class Visitor>
static int64_t visit_ray_voxels(
    double voxel, const double origin[3], const double endpoint[3],
    double length, Visitor&& visitor) {
  const double delta[3] = {
      endpoint[0] - origin[0], endpoint[1] - origin[1],
      endpoint[2] - origin[2]};
  const double full_length = std::sqrt(
      delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
  if (!(length > 0.0) || !(full_length > 1.0e-12)) return 0;
  const double direction[3] = {
      delta[0] / full_length, delta[1] / full_length,
      delta[2] / full_length};
  const double finish[3] = {
      origin[0] + length * direction[0],
      origin[1] + length * direction[1],
      origin[2] + length * direction[2]};
  int32_t current[3], target[3], step[3];
  double next_distance[3], distance_increment[3];
  for (int axis = 0; axis < 3; ++axis) {
    current[axis] = static_cast<int32_t>(std::floor(origin[axis] / voxel));
    target[axis] = static_cast<int32_t>(std::floor(finish[axis] / voxel));
    if (direction[axis] > 1.0e-15) {
      step[axis] = 1;
      const double boundary = static_cast<double>(current[axis] + 1) * voxel;
      next_distance[axis] = (boundary - origin[axis]) / direction[axis];
      distance_increment[axis] = voxel / direction[axis];
    } else if (direction[axis] < -1.0e-15) {
      step[axis] = -1;
      const double boundary = static_cast<double>(current[axis]) * voxel;
      next_distance[axis] = (boundary - origin[axis]) / direction[axis];
      distance_increment[axis] = -voxel / direction[axis];
    } else {
      step[axis] = 0;
      next_distance[axis] = std::numeric_limits<double>::infinity();
      distance_increment[axis] = std::numeric_limits<double>::infinity();
    }
  }
  int64_t visited = 0;
  visitor(pack_voxel(current[0], current[1], current[2]));
  ++visited;
  while (current[0] != target[0] || current[1] != target[1] ||
         current[2] != target[2]) {
    int axis = 0;
    if (next_distance[1] < next_distance[axis]) axis = 1;
    if (next_distance[2] < next_distance[axis]) axis = 2;
    if (next_distance[axis] > length + 1.0e-12) break;
    current[axis] += step[axis];
    next_distance[axis] += distance_increment[axis];
    visitor(pack_voxel(current[0], current[1], current[2]));
    ++visited;
  }
  return visited;
}

// stats: frame, rays, free updates, occupied updates, free voxels,
// occupied voxels.  Input is one origin, endpoint and conservative radius per
// depth return; no unobserved ray or future frame is inserted.
MVT_API int32_t occ_incremental_integrate(
    void* handle, const double* origins, const double* endpoints,
    const double* radii, const int8_t* endpoint_is_occupied,
    int32_t ray_count, int64_t* stats) {
  auto* map = static_cast<NativeIncrementalOccupancy*>(handle);
  if (!map || !origins || !endpoints || !radii || !endpoint_is_occupied ||
      ray_count < 0 || !stats)
    return -1;
  int64_t free_updates = 0;
  int64_t occupied_updates = 0;
  for (int32_t ray = 0; ray < ray_count; ++ray) {
    const double* origin = origins + 3 * ray;
    const double* endpoint = endpoints + 3 * ray;
    const double dx = endpoint[0] - origin[0];
    const double dy = endpoint[1] - origin[1];
    const double dz = endpoint[2] - origin[2];
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double free_length = std::max(
        0.0, distance - radii[ray] - 0.5 * map->voxel);
    if (free_length > 0.0 && distance > 1.0e-12) {
      const int32_t count = std::max<int32_t>(
          1, static_cast<int32_t>(std::ceil(
                 free_length / (0.45 * map->voxel))));
      uint64_t previous = 0;
      bool have_previous = false;
      for (int32_t sample = 0; sample < count; ++sample) {
        const double t = free_length * static_cast<double>(sample) /
                         static_cast<double>(count);
        const int32_t x = static_cast<int32_t>(std::floor(
            (origin[0] + t * dx / distance) / map->voxel));
        const int32_t y = static_cast<int32_t>(std::floor(
            (origin[1] + t * dy / distance) / map->voxel));
        const int32_t z = static_cast<int32_t>(std::floor(
            (origin[2] + t * dz / distance) / map->voxel));
        const uint64_t key = pack_voxel(x, y, z);
        if (have_previous && key == previous) continue;
        incremental_add_score(map, key, -map->free_decrement);
        previous = key;
        have_previous = true;
        ++free_updates;
      }
    }
    if (endpoint_is_occupied[ray]) {
      occupied_updates += visit_conservative_ball(
          map->voxel, endpoint, radii[ray], [&](uint64_t key) {
            incremental_add_score(map, key, map->occupied_increment);
          });
    }
  }
  ++map->frame_index;
  stats[0] = map->frame_index;
  stats[1] = ray_count;
  stats[2] = free_updates;
  stats[3] = occupied_updates;
  stats[4] = map->sparse_free_voxels;
  stats[5] = map->sparse_occupied_voxels;
  return 0;
}

// Directional counterpart of occ_incremental_integrate.  Free ray prefixes
// still use the conservative outer radius, while occupied endpoints insert
// precisely the voxels intersected by the certified sample ellipsoid.  This
// avoids silently replacing a thin pixel/depth certificate by its largest-
// axis sphere in the free/occupied/unknown map.
MVT_API int32_t occ_incremental_integrate_ellipsoids(
    void* handle, const double* origins, const double* endpoints,
    const double* outer_radii, const double* shapes,
    const int8_t* endpoint_is_occupied, int32_t ray_count, int64_t* stats) {
  auto* map = static_cast<NativeIncrementalOccupancy*>(handle);
  if (!map || !origins || !endpoints || !outer_radii || !shapes ||
      !endpoint_is_occupied || ray_count < 0 || !stats)
    return -1;
  int64_t free_updates = 0;
  int64_t occupied_updates = 0;
  for (int32_t ray = 0; ray < ray_count; ++ray) {
    const double* origin = origins + 3 * ray;
    const double* endpoint = endpoints + 3 * ray;
    const double dx = endpoint[0] - origin[0];
    const double dy = endpoint[1] - origin[1];
    const double dz = endpoint[2] - origin[2];
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double free_length = std::max(
        0.0, distance - outer_radii[ray] - 0.5 * map->voxel);
    if (free_length > 0.0 && distance > 1.0e-12) {
      free_updates += visit_ray_voxels(
          map->voxel, origin, endpoint, free_length, [&](uint64_t key) {
        incremental_add_score(map, key, -map->free_decrement);
          });
    }
    if (endpoint_is_occupied[ray]) {
      occupied_updates += visit_conservative_ellipsoid(
          map->voxel, endpoint, shapes + 9 * ray, [&](uint64_t key) {
            incremental_add_score(map, key, map->occupied_increment);
          });
    }
  }
  ++map->frame_index;
  stats[0] = map->frame_index;
  stats[1] = ray_count;
  stats[2] = free_updates;
  stats[3] = occupied_updates;
  stats[4] = map->sparse_free_voxels;
  stats[5] = map->sparse_occupied_voxels;
  return 0;
}

MVT_API int32_t occ_incremental_mark_free(
    void* handle, const double* centers, const double* radii,
    int32_t sphere_count, double padding) {
  auto* map = static_cast<NativeIncrementalOccupancy*>(handle);
  if (!map || !centers || !radii || sphere_count < 0) return -1;
  for (int32_t sphere = 0; sphere < sphere_count; ++sphere) {
    visit_conservative_ball(
        map->voxel, centers + 3 * sphere, radii[sphere] + padding,
        [&](uint64_t key) {
          if (incremental_state(map, key) != 2)
            incremental_set_score(map, key, map->minimum_score);
        });
  }
  return 0;
}

// Direct control-cycle query on the live causal map.  This avoids exporting
// and rebuilding an unordered map at 50 Hz; it uses the same conservative
// swept-ball voxel test and the same free/occupied/unknown thresholds as the
// immutable snapshot oracle below.
MVT_API int32_t occ_incremental_certify_swept_spheres(
    void* handle, const float* starts, const float* ends, const float* radii,
    int32_t sphere_count, float margin, int32_t* counts) {
  auto* map = static_cast<NativeIncrementalOccupancy*>(handle);
  if (!map || !starts || !ends || !radii || sphere_count < 0 || !counts)
    return -1;
  int32_t occupied = 0;
  int32_t unknown = 0;
  std::unordered_set<uint64_t> checked;
  checked.reserve(static_cast<size_t>(sphere_count) * 512);
  const float voxel = static_cast<float>(map->voxel);
  const float half_diagonal = 0.8660254037844386f * voxel;
  for (int32_t sphere = 0; sphere < sphere_count; ++sphere) {
    const float dx = ends[3 * sphere] - starts[3 * sphere];
    const float dy = ends[3 * sphere + 1] - starts[3 * sphere + 1];
    const float dz = ends[3 * sphere + 2] - starts[3 * sphere + 2];
    const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    const int steps = std::max(
        1, static_cast<int>(std::ceil(distance / (0.45f * voxel))));
    const float inflated = std::max(0.0f, radii[sphere] + margin);
    const float cover = inflated + half_diagonal;
    const float cover2 = cover * cover;
    for (int step = 0; step <= steps; ++step) {
      const float alpha = static_cast<float>(step) / static_cast<float>(steps);
      const float center[3] = {
          starts[3 * sphere] + alpha * dx,
          starts[3 * sphere + 1] + alpha * dy,
          starts[3 * sphere + 2] + alpha * dz};
      int32_t lo[3], hi[3];
      for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = static_cast<int32_t>(
            std::floor((center[axis] - cover) / voxel));
        hi[axis] = static_cast<int32_t>(
            std::floor((center[axis] + cover) / voxel));
      }
      for (int32_t x = lo[0]; x <= hi[0]; ++x) {
        const float vx = (static_cast<float>(x) + 0.5f) * voxel - center[0];
        for (int32_t y = lo[1]; y <= hi[1]; ++y) {
          const float vy = (static_cast<float>(y) + 0.5f) * voxel - center[1];
          for (int32_t z = lo[2]; z <= hi[2]; ++z) {
            const float vz =
                (static_cast<float>(z) + 0.5f) * voxel - center[2];
            if (vx * vx + vy * vy + vz * vz > cover2) continue;
            const uint64_t key = pack_voxel(x, y, z);
            if (!checked.insert(key).second) continue;
            const int8_t state = incremental_state(map, key);
            if (state == 2)
              ++occupied;
            else if (state != 1)
              ++unknown;
          }
        }
      }
    }
  }
  counts[0] = occupied;
  counts[1] = unknown;
  counts[2] = static_cast<int32_t>(checked.size());
  return occupied == 0 && unknown == 0 ? 1 : 0;
}

MVT_API int32_t occ_incremental_size(void* handle) {
  auto* map = static_cast<NativeIncrementalOccupancy*>(handle);
  if (!map || map->scores.size() > static_cast<size_t>(INT32_MAX)) return -1;
  return static_cast<int32_t>(map->scores.size());
}

// O(n) immutable snapshot for asynchronous control queries.  Formal online
// runs do not install a calibrated free AABB; rejecting that optional mode
// prevents silently dropping its implicit cells from the clone.
MVT_API void* occ_incremental_clone_snapshot(void* handle) {
  auto* source = static_cast<NativeIncrementalOccupancy*>(handle);
  if (!source || source->has_calibrated_free_aabb) return nullptr;
  try {
    auto* snapshot = new NativeOccupancy();
    snapshot->voxel = static_cast<float>(source->voxel);
    snapshot->states.reserve(source->scores.size());
    for (const auto& item : source->scores) {
      const int8_t state = incremental_score_state(source, item.second);
      if (state != 0) snapshot->states[item.first] = static_cast<uint8_t>(state);
    }
    return snapshot;
  } catch (...) {
    return nullptr;
  }
}

MVT_API int32_t occ_incremental_export(
    void* handle, int32_t* keys, int8_t* states, int32_t capacity) {
  auto* map = static_cast<NativeIncrementalOccupancy*>(handle);
  if (!map || !keys || !states || capacity < 0 ||
      map->scores.size() > static_cast<size_t>(capacity)) return -1;
  std::vector<uint64_t> ordered;
  ordered.reserve(map->scores.size());
  for (const auto& item : map->scores) ordered.push_back(item.first);
  std::sort(ordered.begin(), ordered.end(), [](uint64_t a, uint64_t b) {
    for (int axis = 0; axis < 3; ++axis) {
      const int32_t av = unpack_voxel_axis(a, axis);
      const int32_t bv = unpack_voxel_axis(b, axis);
      if (av != bv) return av < bv;
    }
    return false;
  });
  for (int32_t i = 0; i < static_cast<int32_t>(ordered.size()); ++i) {
    for (int axis = 0; axis < 3; ++axis)
      keys[3 * i + axis] = unpack_voxel_axis(ordered[i], axis);
    states[i] = incremental_state(map, ordered[i]);
  }
  return static_cast<int32_t>(ordered.size());
}

MVT_API int32_t occ_incremental_dirty_size(void* handle) {
  auto* map = static_cast<NativeIncrementalOccupancy*>(handle);
  if (!map || map->dirty_state_keys.size() > static_cast<size_t>(INT32_MAX))
    return -1;
  return static_cast<int32_t>(map->dirty_state_keys.size());
}

// Export only keys whose discrete three-state value changed since the previous
// delta export.  State 0 is included so replay can remove a formerly known
// voxel.  Sorting makes the packet and its hash deterministic.
MVT_API int32_t occ_incremental_export_dirty(
    void* handle, int32_t* keys, int8_t* states, int32_t capacity) {
  auto* map = static_cast<NativeIncrementalOccupancy*>(handle);
  if (!map || !keys || !states || capacity < 0 ||
      map->dirty_state_keys.size() > static_cast<size_t>(capacity)) return -1;
  std::vector<uint64_t> ordered(
      map->dirty_state_keys.begin(), map->dirty_state_keys.end());
  std::sort(ordered.begin(), ordered.end(), [](uint64_t a, uint64_t b) {
    for (int axis = 0; axis < 3; ++axis) {
      const int32_t av = unpack_voxel_axis(a, axis);
      const int32_t bv = unpack_voxel_axis(b, axis);
      if (av != bv) return av < bv;
    }
    return false;
  });
  for (int32_t i = 0; i < static_cast<int32_t>(ordered.size()); ++i) {
    for (int axis = 0; axis < 3; ++axis)
      keys[3 * i + axis] = unpack_voxel_axis(ordered[i], axis);
    states[i] = incremental_state(map, ordered[i]);
  }
  map->dirty_state_keys.clear();
  return static_cast<int32_t>(ordered.size());
}

MVT_API void* occ_create(const int32_t* keys, const int8_t* states, int32_t count, float voxel_size) {
  if (!keys || !states || count < 0 || !(voxel_size > 0.0f)) return nullptr;
  try {
    auto* map = new NativeOccupancy();
    map->voxel = voxel_size;
    map->states.reserve(static_cast<size_t>(count) * 2);
    for (int32_t i = 0; i < count; ++i) {
      if (states[i] == 1 || states[i] == 2) {
        map->states[pack_voxel(keys[3*i], keys[3*i+1], keys[3*i+2])] =
            static_cast<uint8_t>(states[i]);
      }
    }
    return map;
  } catch (...) {
    return nullptr;
  }
}

MVT_API void occ_destroy(void* handle) {
  delete static_cast<NativeOccupancy*>(handle);
}

MVT_API int32_t occ_certify_swept_spheres(
    void* handle,
    const float* starts,
    const float* ends,
    const float* radii,
    int32_t sphere_count,
    float margin,
    int32_t* counts) {
  auto* map = static_cast<NativeOccupancy*>(handle);
  if (!map || !starts || !ends || !radii || sphere_count < 0 || !counts) return -1;
  int32_t occupied = 0;
  int32_t unknown = 0;
  std::unordered_set<uint64_t> checked;
  checked.reserve(static_cast<size_t>(sphere_count) * 512);
  const float voxel = map->voxel;
  const float half_diagonal = 0.8660254037844386f * voxel;
  for (int32_t sphere = 0; sphere < sphere_count; ++sphere) {
    const float dx = ends[3*sphere] - starts[3*sphere];
    const float dy = ends[3*sphere+1] - starts[3*sphere+1];
    const float dz = ends[3*sphere+2] - starts[3*sphere+2];
    const float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    const int steps = std::max(1, static_cast<int>(std::ceil(distance / (0.45f * voxel))));
    const float inflated = std::max(0.0f, radii[sphere] + margin);
    const float cover = inflated + half_diagonal;
    const float cover2 = cover * cover;
    for (int step = 0; step <= steps; ++step) {
      const float alpha = static_cast<float>(step) / static_cast<float>(steps);
      const float center[3] = {
          starts[3*sphere] + alpha*dx,
          starts[3*sphere+1] + alpha*dy,
          starts[3*sphere+2] + alpha*dz};
      int32_t lo[3], hi[3];
      for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = static_cast<int32_t>(std::floor((center[axis] - cover) / voxel));
        hi[axis] = static_cast<int32_t>(std::floor((center[axis] + cover) / voxel));
      }
      for (int32_t x = lo[0]; x <= hi[0]; ++x) {
        const float vx = (static_cast<float>(x) + 0.5f) * voxel - center[0];
        for (int32_t y = lo[1]; y <= hi[1]; ++y) {
          const float vy = (static_cast<float>(y) + 0.5f) * voxel - center[1];
          for (int32_t z = lo[2]; z <= hi[2]; ++z) {
            const float vz = (static_cast<float>(z) + 0.5f) * voxel - center[2];
            if (vx*vx + vy*vy + vz*vz > cover2) continue;
            const uint64_t key = pack_voxel(x, y, z);
            if (!checked.insert(key).second) continue;
            const auto found = map->states.find(key);
            if (found == map->states.end()) {
              ++unknown;
            } else if (found->second == 2) {
              ++occupied;
            } else if (found->second != 1) {
              ++unknown;
            }
          }
        }
      }
    }
  }
  counts[0] = occupied;
  counts[1] = unknown;
  counts[2] = static_cast<int32_t>(checked.size());
  return occupied == 0 && unknown == 0 ? 1 : 0;
}

MVT_API int32_t occ_certify_ellipsoids(
    void* handle, const double* centers, const double* shapes,
    int32_t ellipsoid_count, int32_t* counts) {
  auto* map = static_cast<NativeOccupancy*>(handle);
  if (!map || !centers || !shapes || ellipsoid_count < 0 || !counts) return -1;
  int32_t occupied=0, unknown=0;
  std::unordered_set<uint64_t> checked;
  checked.reserve(static_cast<size_t>(ellipsoid_count)*256);
  const double voxel=map->voxel;
  for (int32_t item=0; item<ellipsoid_count; ++item) {
    const double* center=centers+3*item;
    const double* shape=shapes+9*item;
    int32_t lo[3],hi[3];
    for (int axis=0; axis<3; ++axis) {
      const double half=std::sqrt(std::max(shape[3*axis+axis],0.0));
      lo[axis]=static_cast<int32_t>(std::floor((center[axis]-half)/voxel));
      hi[axis]=static_cast<int32_t>(std::floor((center[axis]+half)/voxel));
    }
    for (int32_t x=lo[0]; x<=hi[0]; ++x)
      for (int32_t y=lo[1]; y<=hi[1]; ++y)
        for (int32_t z=lo[2]; z<=hi[2]; ++z) {
          const uint64_t key=pack_voxel(x,y,z);
          if (checked.find(key)!=checked.end()) continue;
          const double box_lo[3]={x*voxel,y*voxel,z*voxel};
          const double box_hi[3]={(x+1)*voxel,(y+1)*voxel,(z+1)*voxel};
          if (!ellipsoid_intersects_box(center,shape,box_lo,box_hi)) continue;
          checked.insert(key);
          const auto found=map->states.find(key);
          if (found==map->states.end()) ++unknown;
          else if (found->second==2) ++occupied;
          else if (found->second!=1) ++unknown;
        }
  }
  counts[0]=occupied;
  counts[1]=unknown;
  counts[2]=static_cast<int32_t>(checked.size());
  return occupied==0 && unknown==0 ? 1 : 0;
}
