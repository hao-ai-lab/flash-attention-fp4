// Does warp-level mma.sync (legacy HMMA path) overlap with tcgen05.mma (UMMA)?
// Motivation: a tcgen05.mma with M=64 costs the same as M=128 on GB300, so half
// the tensor core looks idle. If mma.sync issued from otherwise-busy softmax
// warps used a different datapath, that idle half could be harvested.
//
// Warp 0        : N_TC x tcgen05.mma (M x N x K16 bf16), commit + mbarrier wait.
// Warps 4..4+W-1: register-resident mma.sync.m16n8k16 chains (NACC independent
//                 accumulators per thread -> ILP, no memory traffic in the loop).
// Modes: tc-only / sync-only / both (started together via __syncthreads).
// If the two paths are independent: both ~= max(alone).  If they share the
// tensor core: both ~= sum(alone).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
  uint64_t d = (uint64_t)((saddr & 0x3FFFF) >> 4); d |= (uint64_t)8 << 16; d |= (uint64_t)16 << 32; d |= (uint64_t)1 << 46; return d; }
__device__ __forceinline__ uint32_t idesc_bf16(int M, int N) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24); }
__device__ __forceinline__ void tcmma(uint32_t taddr, uint64_t ad, uint64_t bd, uint32_t id) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, 1, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
               :: "r"(taddr), "l"(ad), "l"(bd), "r"(id)); }
__device__ __forceinline__ void commit(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) {
  asm volatile("{\n\t.reg .pred P1;\n\tLW:\n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, 10000000;\n\t@P1 bra DN;\n\tbra LW;\n\tDN:\n\t}\n"
               :: "r"(mbar), "r"(phase) : "memory"); }
__device__ __forceinline__ uint64_t clk() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }

// One warp-wide mma.sync.m16n8k16 = 2048 MACs.
#define MMA_SYNC(d, a, b) \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))

template <int NACC, int NTC>
__global__ void __launch_bounds__(256, 1)
bench(uint64_t* out, int mode, int M, int N, int iters, int nsync_warps, int reps) {
  __shared__ __align__(1024) uint16_t sA[128 * 16];
  __shared__ __align__(1024) uint16_t sB[256 * 16];
  __shared__ __align__(8) uint64_t mbar;
  __shared__ uint32_t tmem_base;
  __shared__ uint64_t t_tc, t_sync;
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  for (int i = tid; i < 128 * 16; i += 256) sA[i] = (uint16_t)(0x3C00 + (i * 37 % 64));
  for (int i = tid; i < 256 * 16; i += 256) sB[i] = (uint16_t)(0x3C00 + (i * 53 % 64));
  if (tid == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&mbar)));
                  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); t_tc = 0; t_sync = 0; }
  __syncthreads();
  if (warp == 0) { asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(smem_u32(&tmem_base)));
                   asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  uint32_t taddr = tmem_base, mb = smem_u32(&mbar);
  uint64_t ad = make_desc(smem_u32(sA)), bd = make_desc(smem_u32(sB));
  uint32_t id = idesc_bf16(M, N);
  uint32_t phase = 0;
  uint64_t tot_tc = 0, tot_sync = 0, tot_wall = 0;

  // register-resident operands for mma.sync
  uint32_t ra[4] = {0x3C003C00u, 0x3C003C00u, 0x3C003C00u, 0x3C003C00u};
  uint32_t rb[2] = {0x3C003C00u, 0x3C003C00u};
  float acc[NACC * 4];
#pragma unroll
  for (int i = 0; i < NACC * 4; i++) acc[i] = (float)(lane + i) * 1e-4f;

  bool do_tc = (mode == 0 || mode == 2), do_sync = (mode == 1 || mode == 2);
  bool is_sync_warp = (warp >= 4 && warp < 4 + nsync_warps);

  for (int r = 0; r < reps; r++) {
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    uint64_t t0 = clk();
    if (do_tc && warp == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < NTC; i++) tcmma(taddr, ad, bd, id);
      commit(mb); mbar_wait(mb, phase);
      t_tc = clk();
    }
    if (do_sync && is_sync_warp) {
      for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < NACC; j++) { float* d = acc + j * 4; MMA_SYNC(d, ra, rb); }
      }
      if (lane == 0 && warp == 4) t_sync = clk();
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (tid == 0 && r > 0) {
      uint64_t e_tc = do_tc ? t_tc - t0 : 0, e_sy = do_sync ? t_sync - t0 : 0;
      tot_tc += e_tc; tot_sync += e_sy; tot_wall += (e_tc > e_sy ? e_tc : e_sy);
    }
    phase ^= 1;
    __syncthreads();
  }
  if (tid == 0) { out[0] = tot_tc / (reps - 1); out[1] = tot_sync / (reps - 1); out[2] = tot_wall / (reps - 1); }
  // keep acc alive
  float s = 0;
#pragma unroll
  for (int i = 0; i < NACC * 4; i++) s += acc[i];
  if (s == 12345.678f) out[3] = 1;
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(taddr));
}

struct R { uint64_t tc, sy, wall; };
template <int NACC, int NTC>
static R run(int mode, int M, int N, int iters, int w, int reps) {
  uint64_t* d; CK(cudaMalloc(&d, 32)); CK(cudaMemset(d, 0, 32));
  bench<NACC, NTC><<<1, 256>>>(d, mode, M, N, iters, w, reps);
  CK(cudaDeviceSynchronize());
  uint64_t h[4]; CK(cudaMemcpy(h, d, 32, cudaMemcpyDeviceToHost)); CK(cudaFree(d));
  return {h[0], h[1], h[2]};
}

int main(int argc, char** argv) {
  const int NTC = 32, reps = 40;
  int W = argc > 1 ? atoi(argv[1]) : 4;          // number of mma.sync warps
  int iters = argc > 2 ? atoi(argv[2]) : 64;     // mma.sync iterations per warp
  int nacc = argc > 3 ? atoi(argv[3]) : 4;       // independent accumulators (ILP)
  if (nacc == 8) {
    const int NACC = 8;
    double sm = (double)W * iters * NACC * 2048.0;
    R b = run<NACC, NTC>(1, 128, 256, iters, W, reps);
    R a = run<NACC, NTC>(0, 128, 256, iters, W, reps);
    R c = run<NACC, NTC>(2, 128, 256, iters, W, reps);
    double tm = (double)NTC * 128 * 256 * 16.0;
    printf("NACC=8 W=%d iters=%d: mma.sync alone %llu cy (%.0f MAC/cy) | tcgen05 alone %llu cy (%.0f MAC/cy)\n",
           W, iters, (unsigned long long)b.sy, sm / b.sy, (unsigned long long)a.tc, tm / a.tc);
    printf("  both wall %llu cy -> %.2fx of sum(%llu), aggregate %.0f MAC/cy (%.2fx of tcgen05 alone)\n",
           (unsigned long long)c.wall, c.wall / (double)(a.tc + b.sy), (unsigned long long)(a.tc + b.sy),
           (tm + sm) / c.wall, ((tm + sm) / c.wall) / (tm / a.tc));
    return 0;
  }
  const int NACC = 4;
  printf("tcgen05 stream = %d x tcgen05.mma(MxNxK16 bf16); mma.sync = %d warps x %d iters x %d acc"
         " (m16n8k16)\n", NTC, W, iters, NACC);
  double sync_macs = (double)W * iters * NACC * 2048.0;
  for (int cfg = 0; cfg < 2; cfg++) {
    int M = cfg ? 64 : 128, N = cfg ? 128 : 256;
    double tc_macs = (double)NTC * M * N * 16.0;
    R a = run<NACC, NTC>(0, M, N, iters, W, reps);   // tc only
    R b = run<NACC, NTC>(1, M, N, iters, W, reps);   // sync only
    R c = run<NACC, NTC>(2, M, N, iters, W, reps);   // both
    printf("\n--- tcgen05 tile M%d N%d  (%.0f MACs)   mma.sync total %.0f MACs\n", M, N, tc_macs, sync_macs);
    printf("  tcgen05 alone        : %5llu cy   (%.0f MAC/cy)\n", (unsigned long long)a.tc, tc_macs / a.tc);
    printf("  mma.sync alone       : %5llu cy   (%.0f MAC/cy)\n", (unsigned long long)b.sy, sync_macs / b.sy);
    printf("  both: tcgen05 %5llu cy, mma.sync %5llu cy, wall %5llu cy\n",
           (unsigned long long)c.tc, (unsigned long long)c.sy, (unsigned long long)c.wall);
    printf("    wall vs max(alone)=%llu -> %.2fx    vs sum(alone)=%llu -> %.2fx\n",
           (unsigned long long)(a.tc > b.sy ? a.tc : b.sy), c.wall / (double)(a.tc > b.sy ? a.tc : b.sy),
           (unsigned long long)(a.tc + b.sy), c.wall / (double)(a.tc + b.sy));
    printf("    aggregate: %.0f MAC/cy both vs %.0f tcgen05-alone  -> %.2fx\n",
           (tc_macs + sync_macs) / c.wall, tc_macs / a.tc, ((tc_macs + sync_macs) / c.wall) / (tc_macs / a.tc));
  }
  return 0;
}
