// tcgen05.mma overlap microbenchmark v2: two issuing WARPS.
// Warp 0 issues NL x large (M128 N256 K16) into tmem cols [0,256), commit -> mbar0.
// Warp 1 issues NS x small (M128 Nsmall K16) into tmem cols [256,..), commit -> mbar1.
// Both start together (barrier), wall time = max(end0, end1) - start. Compare with each stream alone.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
  uint64_t d = (uint64_t)((saddr & 0x3FFFF) >> 4); d |= (uint64_t)8 << 16; d |= (uint64_t)16 << 32; d |= (uint64_t)1 << 46; return d; }
__device__ __forceinline__ uint32_t idesc_bf16(int N) { return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | (8u << 24); }
__device__ __forceinline__ void mma(uint32_t taddr, uint64_t adesc, uint64_t bdesc, uint32_t idesc) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, 1, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" :: "r"(taddr), "l"(adesc), "l"(bdesc), "r"(idesc)); }
__device__ __forceinline__ void commit(uint32_t mbar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) {
  asm volatile("{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, 10000000;\n\t@P1 bra DONE;\n\tbra LAB_WAIT;\n\tDONE:\n\t}\n" :: "r"(mbar), "r"(phase) : "memory"); }
__device__ __forceinline__ uint64_t clk() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }

template <int NI>
__device__ __forceinline__ void issue_n(uint32_t taddr, uint64_t adesc, uint64_t bdesc, uint32_t idesc) {
#pragma unroll
  for (int i = 0; i < NI; i++) mma(taddr, adesc, bdesc, idesc);
}

// mode 0: warp0 large only; 1: warp1 small only; 2: both concurrently; 3: both warps large (NL each, separate acc)
template <int NL, int NS, int NSMALL>
__global__ void __launch_bounds__(128, 1) bench(uint64_t* out, int mode, int reps) {
  __shared__ __align__(1024) uint16_t sA[128 * 16];
  __shared__ __align__(1024) uint16_t sB[256 * 16];
  __shared__ __align__(8) uint64_t mbar[2];
  __shared__ uint32_t tmem_base;
  __shared__ uint64_t t_end[2];
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  for (int i = tid; i < 128 * 16; i += 128) sA[i] = (uint16_t)(0x3C00 + (i * 37 % 64));
  for (int i = tid; i < 256 * 16; i += 128) sB[i] = (uint16_t)(0x3C00 + (i * 53 % 64));
  if (tid == 0) {
    for (int i = 0; i < 2; i++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&mbar[i])));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(smem_u32(&tmem_base)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  uint32_t taddr = tmem_base;
  uint64_t adesc = make_desc(smem_u32(sA)), bdesc = make_desc(smem_u32(sB));
  uint32_t idL = idesc_bf16(256), idS = idesc_bf16(NSMALL);
  uint32_t phase = 0; uint64_t total = 0;
  for (int r = 0; r < reps; r++) {
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    uint64_t t0 = clk();
    if (lane == 0) {
      if (warp == 0 && (mode == 0 || mode == 2 || mode == 3)) {
        issue_n<NL>(taddr, adesc, bdesc, idL); commit(smem_u32(&mbar[0])); mbar_wait(smem_u32(&mbar[0]), phase); t_end[0] = clk();
      }
      if (warp == 1 && (mode == 1 || mode == 2)) {
        issue_n<NS>(taddr + 256, adesc, bdesc, idS); commit(smem_u32(&mbar[1])); mbar_wait(smem_u32(&mbar[1]), phase); t_end[1] = clk();
      }
      if (warp == 1 && mode == 3) {
        issue_n<NL>(taddr + 256, adesc, bdesc, idL); commit(smem_u32(&mbar[1])); mbar_wait(smem_u32(&mbar[1]), phase); t_end[1] = clk();
      }
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
      uint64_t e0 = (mode == 1) ? 0 : t_end[0], e1 = (mode == 0) ? 0 : t_end[1];
      uint64_t e = e0 > e1 ? e0 : e1;
      if (r > 0) total += e - t0;
    }
    phase ^= (mode == 0 && warp == 0) || (mode == 1 && warp == 1) || mode >= 2 ? 1 : 0;
  }
  __syncthreads();
  if (tid == 0) out[mode] = total / (reps - 1);
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(taddr));
}

int main(int argc, char** argv) {
  int nsmall = argc > 1 ? atoi(argv[1]) : 32;
  int reps = 50;
  uint64_t* d; CK(cudaMalloc(&d, 8 * sizeof(uint64_t))); CK(cudaMemset(d, 0, 64));
  for (int mode = 0; mode < 4; mode++) {
    if (nsmall == 32) bench<32, 32, 32><<<1, 128>>>(d, mode, reps);
    else if (nsmall == 64) bench<32, 32, 64><<<1, 128>>>(d, mode, reps);
    else bench<32, 32, 128><<<1, 128>>>(d, mode, reps);
    CK(cudaDeviceSynchronize());
  }
  uint64_t h[8]; CK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  printf("two-warp issue, nL=32 (N256) nS=32 (N%d), cycles wall:\n", nsmall);
  printf("  warp0 large alone       : %llu\n", (unsigned long long)h[0]);
  printf("  warp1 small alone       : %llu\n", (unsigned long long)h[1]);
  printf("  both concurrently       : %llu   (sum %llu, max %llu)  -> %.2fx of sum\n", (unsigned long long)h[2], (unsigned long long)(h[0] + h[1]), (unsigned long long)(h[0] > h[1] ? h[0] : h[1]), (double)h[2] / (double)(h[0] + h[1]));
  printf("  both warps large (2x32) : %llu   (2x alone = %llu) -> %.2fx\n", (unsigned long long)h[3], (unsigned long long)(2 * h[0]), (double)h[3] / (double)(2 * h[0]));
  return 0;
}
