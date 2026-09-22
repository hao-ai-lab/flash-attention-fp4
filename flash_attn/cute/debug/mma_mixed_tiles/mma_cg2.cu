// cta_group::2 (2-CTA) tcgen05.mma cost per SM, GB300. Cluster of 2 CTAs; leader (rank 0) issues
// tcgen05.mma.cta_group::2 with M in {128, 256}, N in {64,128,256}, kind::f16 K16; commit multicast to both
// CTAs' mbarriers; both wait. Cycles per instruction measured on the leader. Compare with cta_group::1 M128
// (140 @N256, 80 @N128): with cta_group::2 M=128 each SM computes 128 rows x N/2 (Layout B, 2x2).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) { uint64_t d = (uint64_t)((saddr & 0x3FFFF) >> 4); d |= (uint64_t)8 << 16; d |= (uint64_t)16 << 32; d |= (uint64_t)1 << 46; return d; }
__device__ __forceinline__ uint32_t idesc(int M, int N) { return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24); }
__device__ __forceinline__ uint64_t clk() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory"); }

__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) bench(uint64_t* out, int M, int N, int reps) {
  __shared__ __align__(1024) uint8_t sA[128 * 32]; __shared__ __align__(1024) uint8_t sB[256 * 32];
  __shared__ __align__(8) uint64_t mbar; __shared__ uint32_t tmem_base;
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32; uint32_t rank = cluster_rank();
  for (int i = tid; i < 128 * 32; i += 128) sA[i] = (uint8_t)(0x38 + (i * 7 % 8));
  for (int i = tid; i < 256 * 32; i += 128) sB[i] = (uint8_t)(0x38 + (i * 5 % 8));
  if (tid == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&mbar))); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  if (warp == 0) { asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(smem_u32(&tmem_base)) : "memory"); asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;"); }
  cluster_sync(); asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  uint32_t taddr = tmem_base, mb = smem_u32(&mbar);
  uint64_t ad = make_desc(smem_u32(sA)), bd = make_desc(smem_u32(sB)); uint32_t id = idesc(M, N);
  uint32_t phase = 0; uint64_t total = 0;
  for (int r = 0; r < reps; r++) {
    cluster_sync(); asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    uint64_t t0 = clk();
    if (rank == 0 && warp == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 32; i++)
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, 1, 0;\n\ttcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" :: "r"(taddr), "l"(ad), "l"(bd), "r"(id));
      asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" :: "r"(mb), "h"((uint16_t)3) : "memory");
    }
    if (warp == 0 && lane == 0) {
      asm volatile("{\n\t.reg .pred P1;\n\tLW:\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1, 10000000;\n\t@P1 bra DN;\n\tbra LW;\n\tDN:\n\t}\n" :: "r"(mb), "r"(phase) : "memory");
    }
    uint64_t t1 = clk();
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (rank == 0 && tid == 0 && r > 0) total += t1 - t0;
    phase ^= 1;
  }
  cluster_sync();
  if (rank == 0 && tid == 0) out[0] = total / (reps - 1);
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(taddr));
  cluster_sync();
}
int main() {
  uint64_t* d; CK(cudaMalloc(&d, 8));
  printf("cta_group::2 kind::f16 K16, cycles per instruction (leader clock), 32 back-to-back:\n         N=    64   128   256\n");
  for (int M : {128, 256}) { printf("  M=%3d  ", M); for (int N : {64, 128, 256}) { bench<<<2, 128>>>(d, M, N, 40); CK(cudaDeviceSynchronize()); uint64_t h; CK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost)); printf("%6.1f", h / 32.0); } printf("\n"); }
  printf("(cta_group::1 reference: M128 N64 63, N128 80, N256 141; ws M64 N256 92.7)\n");
  return 0;
}
