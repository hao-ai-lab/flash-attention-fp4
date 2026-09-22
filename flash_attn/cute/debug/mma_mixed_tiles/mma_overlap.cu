// tcgen05.mma overlap microbenchmark (GB300 / sm_103a).
// Question: can tensor-core MMAs of different N tile sizes overlap / be
// reordered so that a mixed sequence finishes faster than the sum of its
// parts? We time (with %clock64) issue + tcgen05.commit + mbarrier wait for:
//   A: nL x large (M128 N256 K16)      B: nS x small (M128 Nsmall K16)
//   C: nL large then nS small          D: interleaved L,S,L,S,...
//   E: like C but small accumulates into the SAME tmem columns as large
//      (true data dependency through D)
// bf16 x bf16 -> f32, A/B operands in smem (SWIZZLE_NONE K-major core
// matrices: 8 rows x 16B contiguous, LBO=128B between K chunks, SBO=256B
// between 8-row groups). Values are irrelevant for timing.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
  // start addr>>4 [0:14) | LBO>>4=8 [16:30) | SBO>>4=16 [32:46) | version=1 [46:48) | layout SWIZZLE_NONE [61:64)
  uint64_t d = (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)8 << 16;
  d |= (uint64_t)16 << 32;
  d |= (uint64_t)1 << 46;
  return d;
}
__device__ __forceinline__ uint32_t idesc_bf16(int N) {
  // c_fmt F32=1 [4:6) | a_fmt BF16=1 [7:10) | b_fmt BF16=1 [10:13) | K-major | n_dim=N>>3 [17:23) | m_dim=128>>4 [24:29)
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | (8u << 24);
}
__device__ __forceinline__ void mma(uint32_t taddr, uint64_t adesc, uint64_t bdesc, uint32_t idesc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, 1, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
      :: "r"(taddr), "l"(adesc), "l"(bdesc), "r"(idesc));
}
__device__ __forceinline__ void commit(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, 10000000;\n\t"
      "@P1 bra DONE;\n\tbra LAB_WAIT;\n\tDONE:\n\t}\n" :: "r"(mbar), "r"(phase) : "memory");
}
__device__ __forceinline__ uint64_t clk() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }

// Compile-time sequence: NL large + NS small; INTER=interleave; SAME=small
// accumulates into the same tmem columns as large. Descriptors precomputed
// in registers, loop fully unrolled -> issue cost is a few cycles per MMA.
template <int NL, int NS, bool INTER, bool SAME>
__device__ __forceinline__ uint64_t run_seq(uint32_t taddr, uint64_t adesc, uint64_t bdesc, uint32_t idL, uint32_t idS,
                                            uint32_t mbar, uint32_t& phase, int reps, uint64_t* issue_cyc) {
  uint64_t total = 0, issue_total = 0;
  const uint32_t colS = SAME ? 0u : 256u;
  for (int r = 0; r < reps; r++) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    uint64_t t0 = clk();
    if (INTER) {
#pragma unroll
      for (int i = 0; i < (NL > NS ? NL : NS); i++) {
        if (i < NL) mma(taddr, adesc, bdesc, idL);
        if (i < NS) mma(taddr + colS, adesc, bdesc, idS);
      }
    } else {
#pragma unroll
      for (int i = 0; i < NL; i++) mma(taddr, adesc, bdesc, idL);
#pragma unroll
      for (int i = 0; i < NS; i++) mma(taddr + colS, adesc, bdesc, idS);
    }
    uint64_t t_issue = clk();
    commit(mbar);
    mbar_wait(mbar, phase);
    phase ^= 1;
    uint64_t t1 = clk();
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    if (r > 0) { total += t1 - t0; issue_total += t_issue - t0; }
  }
  *issue_cyc = issue_total / (reps - 1);
  return total / (reps - 1);
}

template <int NL, int NS, int NSMALL>
__device__ void run_all(uint64_t* out, uint32_t taddr, uint32_t sa, uint32_t sb, uint32_t mb, uint32_t& phase, int reps) {
  uint64_t adesc = make_desc(sa), bdesc = make_desc(sb), ic;
  uint32_t idL = idesc_bf16(256), idS = idesc_bf16(NSMALL);
  out[0] = run_seq<NL, 0, false, false>(taddr, adesc, bdesc, idL, idS, mb, phase, reps, &ic); out[1] = ic;
  out[2] = run_seq<0, NS, false, false>(taddr, adesc, bdesc, idL, idS, mb, phase, reps, &ic); out[3] = ic;
  out[4] = run_seq<NL, NS, false, false>(taddr, adesc, bdesc, idL, idS, mb, phase, reps, &ic); out[5] = ic;
  out[6] = run_seq<NL, NS, true, false>(taddr, adesc, bdesc, idL, idS, mb, phase, reps, &ic); out[7] = ic;
  out[8] = run_seq<NL, NS, false, true>(taddr, adesc, bdesc, idL, idS, mb, phase, reps, &ic); out[9] = ic;
  out[10] = run_seq<NL, NS, true, true>(taddr, adesc, bdesc, idL, idS, mb, phase, reps, &ic); out[11] = ic;
}

__global__ void __launch_bounds__(128, 1) bench(uint64_t* out, int cfg, int reps) {
  __shared__ __align__(1024) uint16_t sA[128 * 16];  // 128 rows x K16 bf16 = 4 KB
  __shared__ __align__(1024) uint16_t sB[256 * 16];  // up to N=256 rows x K16 = 8 KB
  __shared__ __align__(8) uint64_t mbar;
  __shared__ uint32_t tmem_base;
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  for (int i = tid; i < 128 * 16; i += 128) sA[i] = (uint16_t)(0x3C00 + (i * 37 % 64));  // ~1.0..1.9
  for (int i = tid; i < 256 * 16; i += 128) sB[i] = (uint16_t)(0x3C00 + (i * 53 % 64));
  if (tid == 0) {
    uint32_t m = smem_u32(&mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(m));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {
    uint32_t dst = smem_u32(&tmem_base);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(dst));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  uint32_t taddr = tmem_base;
  uint32_t mb = smem_u32(&mbar);
  uint32_t phase = 0;
  if (tid == 0) {
    uint32_t sa = smem_u32(sA), sb = smem_u32(sB);
    switch (cfg) {
      case 0: run_all<32, 32, 32>(out, taddr, sa, sb, mb, phase, reps); break;
      case 1: run_all<32, 32, 64>(out, taddr, sa, sb, mb, phase, reps); break;
      case 2: run_all<32, 32, 128>(out, taddr, sa, sb, mb, phase, reps); break;
      case 3: run_all<16, 64, 16>(out, taddr, sa, sb, mb, phase, reps); break;
      case 4: run_all<8, 8, 32>(out, taddr, sa, sb, mb, phase, reps); break;
      default: run_all<32, 32, 8>(out, taddr, sa, sb, mb, phase, reps); break;
    }
  }
  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(taddr));
  }
}

int main(int argc, char** argv) {
  int cfg = argc > 1 ? atoi(argv[1]) : 0;
  int reps = argc > 2 ? atoi(argv[2]) : 50;
  const int NLs[] = {32, 32, 32, 16, 8, 32}, NSs[] = {32, 32, 32, 64, 8, 32}, NSM[] = {32, 64, 128, 16, 32, 8};
  int nL = NLs[cfg], nS = NSs[cfg], Nsmall = NSM[cfg];
  uint64_t* d; CK(cudaMalloc(&d, 16 * sizeof(uint64_t)));
  bench<<<1, 128>>>(d, cfg, reps);
  CK(cudaDeviceSynchronize());
  uint64_t h[16]; CK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  printf("nL=%d (M128 N256 K16)  nS=%d (M128 N%d K16)  reps=%d   [cycles: total(issue-only)]\n", nL, nS, Nsmall, reps);
  printf("A  all large            : %6llu (%llu)   per-L %.1f\n", (unsigned long long)h[0], (unsigned long long)h[1], (double)h[0] / nL);
  printf("B  all small            : %6llu (%llu)   per-S %.1f\n", (unsigned long long)h[2], (unsigned long long)h[3], (double)h[2] / nS);
  printf("C  L-block then S-block : %6llu (%llu)   vs A+B %llu  -> %.2fx\n", (unsigned long long)h[4], (unsigned long long)h[5], (unsigned long long)(h[0] + h[2]), (double)h[4] / (double)(h[0] + h[2]));
  printf("D  interleaved L,S      : %6llu (%llu)   vs A+B %llu  -> %.2fx   vs C %.2fx\n", (unsigned long long)h[6], (unsigned long long)h[7], (unsigned long long)(h[0] + h[2]), (double)h[6] / (double)(h[0] + h[2]), (double)h[6] / (double)h[4]);
  printf("E  C but same accumulator: %6llu (%llu)\n", (unsigned long long)h[8], (unsigned long long)h[9]);
  printf("F  D but same accumulator: %6llu (%llu)\n", (unsigned long long)h[10], (unsigned long long)h[11]);
  return 0;
}
