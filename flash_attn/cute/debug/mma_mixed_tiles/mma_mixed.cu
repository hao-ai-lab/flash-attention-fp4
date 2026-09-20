// tcgen05.mma mixed-tile microbenchmark (GB300 / sm_103a).
//   ladder : per-instruction cycles for M in {64,128} x N in {8..256}, kind::f16 (K16) and kind::f8f6f4 (K32)
//   mix    : fraction f of "decode-like" small tiles among "prefill-like" large tiles, three schedules
//            (blocked / interleaved / two issuing warps), vs the additive model
//   bg     : the large+small stream while 4 other warps run a MUFU/FMA (softmax-like) loop
// Single CTA of 256 threads (8 warps); warp 0/1 issue MMAs, warps 4-7 are background compute.
// Values are irrelevant for timing. Timing: %clock64 from issue start to mbarrier completion.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
// SWIZZLE_NONE K-major core matrices (8 rows x 16B), LBO=128B, SBO=256B; K16 bf16 / K32 e4m3 both = 32B per row.
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
  uint64_t d = (uint64_t)((saddr & 0x3FFFF) >> 4); d |= (uint64_t)8 << 16; d |= (uint64_t)16 << 32; d |= (uint64_t)1 << 46; return d; }
// c=F32 (1<<4); a/b fmt: BF16=1 for kind::f16, E4M3=0 for kind::f8f6f4; K-major; n_dim [17:23); m_dim [24:29)
__device__ __forceinline__ uint32_t idesc(int M, int N, bool f8) {
  uint32_t a = f8 ? 0u : 1u; return (1u << 4) | (a << 7) | (a << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24); }
template <bool F8> __device__ __forceinline__ void mma(uint32_t taddr, uint64_t adesc, uint64_t bdesc, uint32_t id) {
  if (F8) asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, 1, 0;\n\ttcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}\n" :: "r"(taddr), "l"(adesc), "l"(bdesc), "r"(id));
  else    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, 1, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" :: "r"(taddr), "l"(adesc), "l"(bdesc), "r"(id));
}
__device__ __forceinline__ void commit(uint32_t mbar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) {
  asm volatile("{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, 10000000;\n\t@P1 bra DONE;\n\tbra LAB_WAIT;\n\tDONE:\n\t}\n" :: "r"(mbar), "r"(phase) : "memory"); }
__device__ __forceinline__ uint64_t clk() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }

// Compile-time issue sequences (fully unrolled, no per-instruction branches).
// SCHED 0: blocked (NL large then NS small); SCHED 1: interleaved (small spread evenly).
__host__ __device__ constexpr bool is_large_at(int i, int nL, int nS, int sched) {
  if (sched == 0) return i < nL;
  // interleaved: position i is small if floor((i+1)*nS/(nL+nS)) > floor(i*nS/(nL+nS))
  return ((i + 1) * nS / (nL + nS)) == (i * nS / (nL + nS));
}
template <bool F8, int NL, int NS, int SCHED>
__device__ __forceinline__ void issue_seq(uint32_t taddr, uint32_t colS, uint64_t ad, uint64_t bd, uint32_t idL, uint32_t idS) {
#pragma unroll
  for (int i = 0; i < NL + NS; i++) {
    if (is_large_at(i, NL, NS, SCHED)) mma<F8>(taddr, ad, bd, idL); else mma<F8>(taddr + colS, ad, bd, idS);
  }
}
template <bool F8>
__device__ __forceinline__ void issue_dispatch(int nL, int nS, int sched, uint32_t taddr, uint32_t colS, uint64_t ad, uint64_t bd, uint32_t idL, uint32_t idS) {
#define CASE(L, S) if (nL == L && nS == S) { if (sched == 0) issue_seq<F8, L, S, 0>(taddr, colS, ad, bd, idL, idS); else issue_seq<F8, L, S, 1>(taddr, colS, ad, bd, idL, idS); return; }
  CASE(32, 0) CASE(0, 32) CASE(28, 4) CASE(24, 8) CASE(16, 16) CASE(8, 24) CASE(4, 28)
  CASE(28, 0) CASE(24, 0) CASE(16, 0) CASE(8, 0) CASE(4, 0) CASE(0, 4) CASE(0, 8) CASE(0, 16) CASE(0, 24) CASE(0, 28)
#undef CASE
}

struct Params { int mode; int nL, nS; int ML, NL, MS, NS; int f8; int sched; int reps; int bg; };

__global__ void __launch_bounds__(256, 1) bench(uint64_t* out, Params P) {
  __shared__ __align__(1024) uint8_t sA[128 * 32];   // 128 rows x 32B
  __shared__ __align__(1024) uint8_t sB[256 * 32];   // up to 256 rows x 32B
  __shared__ __align__(8) uint64_t mbar[2];
  __shared__ uint32_t tmem_base;
  __shared__ uint64_t t_end[2];
  __shared__ int stop;
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  for (int i = tid; i < 128 * 32; i += 256) sA[i] = (uint8_t)(0x38 + (i * 7 % 8));
  for (int i = tid; i < 256 * 32; i += 256) sB[i] = (uint8_t)(0x38 + (i * 5 % 8));
  if (tid == 0) { for (int i = 0; i < 2; i++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&mbar[i]))); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); stop = 0; }
  __syncthreads();
  if (warp == 0) { asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(smem_u32(&tmem_base))); asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  uint32_t taddr = tmem_base;
  uint64_t ad = make_desc(smem_u32(sA)), bd = make_desc(smem_u32(sB));
  uint32_t idL = idesc(P.ML, P.NL, P.f8), idS = idesc(P.MS, P.NS, P.f8);
  uint32_t colS = 256;
  uint32_t phase = 0; uint64_t total = 0; float bgacc = 1.0f;
  for (int r = 0; r < P.reps; r++) {
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    uint64_t t0 = clk();
    if (warp >= 4 && P.bg) {
      // softmax-like background: MUFU ex2 + FMA chain until the MMA warps finish
      float x = bgacc + lane * 1e-3f;
      while (!*((volatile int*)&stop)) {
#pragma unroll 8
        for (int i = 0; i < 8; i++) { x = exp2f(x * 0.5f - 1.0f) + 0.25f * x; }
      }
      bgacc = x;
    }
    if (lane == 0 && warp < 2) {
      if (P.mode == 0 && warp == 0) {            // single warp, pattern order
        if (P.f8) issue_dispatch<true>(P.nL, P.nS, P.sched, taddr, colS, ad, bd, idL, idS); else issue_dispatch<false>(P.nL, P.nS, P.sched, taddr, colS, ad, bd, idL, idS);
        commit(smem_u32(&mbar[0])); mbar_wait(smem_u32(&mbar[0]), phase); t_end[0] = clk(); t_end[1] = 0;
      }
      if (P.mode == 1) {                          // two warps: warp0 all large, warp1 all small
        if (warp == 0) { if (P.f8) issue_dispatch<true>(P.nL, 0, 0, taddr, colS, ad, bd, idL, idS); else issue_dispatch<false>(P.nL, 0, 0, taddr, colS, ad, bd, idL, idS);
                         commit(smem_u32(&mbar[0])); mbar_wait(smem_u32(&mbar[0]), phase); t_end[0] = clk(); }
        else           { if (P.f8) issue_dispatch<true>(0, P.nS, 0, taddr, colS, ad, bd, idL, idS); else issue_dispatch<false>(0, P.nS, 0, taddr, colS, ad, bd, idL, idS);
                         commit(smem_u32(&mbar[1])); mbar_wait(smem_u32(&mbar[1]), phase); t_end[1] = clk(); }
      }
    }
    if (P.mode == 0 && warp == 0 && lane == 0) *((volatile int*)&stop) = 1;
    if (P.mode == 1 && warp < 2 && lane == 0) { __threadfence_block(); atomicAdd(&stop, 1); }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (tid == 0) { uint64_t e = t_end[0] > t_end[1] ? t_end[0] : t_end[1]; if (r > 0) total += e - t0; stop = 0; }
    phase ^= 1;
    __syncthreads();
  }
  if (tid == 0) { out[0] = total / (P.reps - 1); out[1] = (uint64_t)(bgacc != 12345.f); }
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(taddr));
}

static uint64_t run(Params P) {
  uint64_t* d; CK(cudaMalloc(&d, 16)); bench<<<1, 256>>>(d, P); CK(cudaDeviceSynchronize());
  uint64_t h[2]; CK(cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost)); CK(cudaFree(d)); return h[0];
}
int main(int argc, char** argv) {
  const char* what = argc > 1 ? argv[1] : "ladder";
  int reps = 40;
  if (!strcmp(what, "ladder")) {
    for (int f8 = 0; f8 < 2; f8++) {
      printf("kind::%s per-instruction cycles (32 back-to-back, same accumulator):\n         N=", f8 ? "f8f6f4 (K32)" : "f16 (K16)");
      int Ns[] = {8, 16, 32, 64, 128, 256}; for (int n : Ns) printf("%6d", n); printf("\n");
      for (int M : {64, 128}) { printf("  M=%3d  ", M); for (int n : Ns) { Params P{0, 32, 0, M, n, M, n, f8, 0, reps, 0}; printf("%6.1f", run(P) / 32.0); } printf("\n"); }
    }
  } else if (!strcmp(what, "mix")) {
    int f8 = argc > 2 ? atoi(argv[2]) : 0; int MS = argc > 3 ? atoi(argv[3]) : 64; int NS = argc > 4 ? atoi(argv[4]) : 128;
    int total = 32;
    Params base{0, 32, 0, 128, 256, MS, NS, f8, 0, reps, 0}; uint64_t L1 = run(base);
    Params bs{0, 0, 32, 128, 256, MS, NS, f8, 0, reps, 0}; uint64_t S1 = run(bs);
    printf("kind::%s large=M128xN256 small=M%dxN%d; per-instr: large %.1f small %.1f cycles\n", f8 ? "f8f6f4" : "f16", MS, NS, L1 / 32.0, S1 / 32.0);
    printf("  f(small)  blocked  interleaved  two-warps   additive-model   blocked/model  inter/model  2warp/model\n");
    for (int nS : {0, 4, 8, 16, 24, 28, 32}) {
      int nL = total - nS;
      uint64_t model = (uint64_t)(nL * (L1 / 32.0) + nS * (S1 / 32.0));
      Params pb{0, nL, nS, 128, 256, MS, NS, f8, 0, reps, 0};   // blocked: large first
      Params pi{0, nL, nS, 128, 256, MS, NS, f8, 1, reps, 0};
      Params pw{1, nL, nS, 128, 256, MS, NS, f8, 0, reps, 0};
      uint64_t b = run(pb), i = run(pi), w = (nL && nS) ? run(pw) : (nL ? b : b);
      printf("  %5.2f   %7llu  %11llu  %9llu   %14llu   %8.2f  %10.2f  %10.2f\n", nS / (double)total, (unsigned long long)b, (unsigned long long)i, (unsigned long long)w, (unsigned long long)model, b / (double)model, i / (double)model, w / (double)model);
    }
  } else if (!strcmp(what, "bg")) {
    for (int f8 = 0; f8 < 2; f8++) for (int bg = 0; bg < 2; bg++) {
      Params pl{0, 32, 0, 128, 256, 64, 128, f8, 0, reps, bg}; Params ps{0, 0, 32, 128, 256, 64, 128, f8, 0, reps, bg};
      Params pm{0, 16, 16, 128, 256, 64, 128, f8, 1, reps, bg};
      printf("kind::%s bg=%d : 32 large %llu | 32 small(M64 N128) %llu | 16+16 interleaved %llu cycles\n", f8 ? "f8f6f4" : "f16", bg, (unsigned long long)run(pl), (unsigned long long)run(ps), (unsigned long long)run(pm));
    }
  }
  return 0;
}
