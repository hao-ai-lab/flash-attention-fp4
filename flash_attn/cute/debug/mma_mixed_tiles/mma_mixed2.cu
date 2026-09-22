// Mixed-tile follow-ups (GB300): hardware mechanisms that could let the tensor core pair/pipeline tiles.
//  lanepair : M=64 (non-.ws, "1/2 datapath", lane alignment 0 or 16) streams at lane 0 only vs alternating lane 0/16
//  ws       : tcgen05.mma.ws ladder for M in {32,64,128} x N in {64,128,256}
//  kinds    : large kind::f16 + small kind::f8f6f4, blocked vs interleaved vs additive
//  asrc     : large A-from-smem + small A-from-tmem, blocked vs interleaved vs additive
//  cta2     : two co-resident CTAs per SM (one large-tile, one small-tile) vs one CTA per SM, kernel wall time
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) { uint64_t d = (uint64_t)((saddr & 0x3FFFF) >> 4); d |= (uint64_t)8 << 16; d |= (uint64_t)16 << 32; d |= (uint64_t)1 << 46; return d; }
__device__ __forceinline__ uint32_t idesc(int M, int N, bool f8) { uint32_t a = f8 ? 0u : 1u; return (1u << 4) | (a << 7) | (a << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24); }
#define MMA_F16(d, a, b, id) asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, 1, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" :: "r"(d), "l"(a), "l"(b), "r"(id))
#define MMA_F8(d, a, b, id)  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, 1, 0;\n\ttcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}\n" :: "r"(d), "l"(a), "l"(b), "r"(id))
#define MMA_TS(d, atm, b, id) asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, 1, 0;\n\ttcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n" :: "r"(d), "r"(atm), "l"(b), "r"(id))
#define MMA_WS(d, a, b, id)  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, 1, 0;\n\ttcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" :: "r"(d), "l"(a), "l"(b), "r"(id))
__device__ __forceinline__ void commit(uint32_t mbar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) { asm volatile("{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, 10000000;\n\t@P1 bra DONE;\n\tbra LAB_WAIT;\n\tDONE:\n\t}\n" :: "r"(mbar), "r"(phase) : "memory"); }
__device__ __forceinline__ uint64_t clk() { uint64_t t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }

struct Ctx { uint32_t taddr, mbar0, mbar1; uint64_t ad, bd; uint32_t phase; };

__device__ __forceinline__ void setup(Ctx& c, uint8_t* sA, uint8_t* sB, uint64_t* mbar, uint32_t* tmem_base, int ncols) {
  int tid = threadIdx.x, warp = tid / 32;
  for (int i = tid; i < 128 * 32; i += blockDim.x) sA[i] = (uint8_t)(0x38 + (i * 7 % 8));
  for (int i = tid; i < 256 * 32; i += blockDim.x) sB[i] = (uint8_t)(0x38 + (i * 5 % 8));
  if (tid == 0) { for (int i = 0; i < 2; i++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&mbar[i]))); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  if (warp == 0) { if (ncols == 512) asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(smem_u32(tmem_base)));
                   else asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" :: "r"(smem_u32(tmem_base)));
                   asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  c.taddr = *tmem_base; c.mbar0 = smem_u32(&mbar[0]); c.mbar1 = smem_u32(&mbar[1]);
  c.ad = make_desc(smem_u32(sA)); c.bd = make_desc(smem_u32(sB)); c.phase = 0;
}

// ---------- single-CTA experiments (mode selects), timed by warp 0 lane 0 (and warp 1 for two-warp variants)
__global__ void __launch_bounds__(128, 1) bench(uint64_t* out, int mode, int reps) {
  __shared__ __align__(1024) uint8_t sA[128 * 32]; __shared__ __align__(1024) uint8_t sB[256 * 32];
  __shared__ __align__(8) uint64_t mbar[2]; __shared__ uint32_t tmem_base; __shared__ uint64_t t_end[2];
  Ctx c; setup(c, sA, sB, mbar, &tmem_base, 512);
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const uint32_t L16 = 16u << 16;  // lane offset 16 in the TMEM address
  uint32_t idM64 = idesc(64, 256, false), idM128 = idesc(128, 256, false);
  uint32_t idL = idesc(128, 256, false), idS8 = idesc(128, 64, true), idS16 = idesc(128, 64, false);
  uint32_t idws32 = idesc(32, 256, false), idws64 = idesc(64, 256, false), idws128 = idesc(128, 256, false);
  uint64_t total = 0;
  for (int r = 0; r < reps; r++) {
    __syncthreads(); asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    uint64_t t0 = clk();
    if (lane == 0 && warp == 0) {
      switch (mode) {
        case 0: { for (int i = 0; i < 32; i++) MMA_F16(c.taddr, c.ad, c.bd, idM128); break; }                       // 32 x M128 N256
        case 1: { for (int i = 0; i < 32; i++) MMA_F16(c.taddr, c.ad, c.bd, idM64); break; }                        // 32 x M64 N256, all at lane 0
        case 2: { for (int i = 0; i < 16; i++) { MMA_F16(c.taddr, c.ad, c.bd, idM64); MMA_F16(c.taddr + L16, c.ad, c.bd, idM64); } break; }  // alternating lane 0 / 16
        case 3: { for (int i = 0; i < 16; i++) MMA_F16(c.taddr, c.ad, c.bd, idM64); for (int i = 0; i < 16; i++) MMA_F16(c.taddr + L16, c.ad, c.bd, idM64); break; } // blocked 0 then 16
        case 4: { for (int i = 0; i < 16; i++) { MMA_F16(c.taddr, c.ad, c.bd, idM64); MMA_F16(c.taddr + 256, c.ad, c.bd, idM64); } break; } // alternating column halves, same lanes
        // ws ladder
        case 10: { for (int i = 0; i < 32; i++) MMA_WS(c.taddr, c.ad, c.bd, idws32); break; }
        case 11: { for (int i = 0; i < 32; i++) MMA_WS(c.taddr, c.ad, c.bd, idws64); break; }
        case 12: { for (int i = 0; i < 32; i++) MMA_WS(c.taddr, c.ad, c.bd, idws128); break; }
        case 13: { for (int i = 0; i < 32; i++) MMA_WS(c.taddr, c.ad, c.bd, idesc(32, 64, false)); break; }
        case 14: { for (int i = 0; i < 32; i++) MMA_WS(c.taddr, c.ad, c.bd, idesc(64, 64, false)); break; }
        case 15: { for (int i = 0; i < 32; i++) MMA_WS(c.taddr, c.ad, c.bd, idesc(128, 64, false)); break; }
        case 16: { for (int i = 0; i < 32; i++) MMA_WS(c.taddr, c.ad, c.bd, idesc(32, 128, false)); break; }
        case 17: { for (int i = 0; i < 32; i++) MMA_WS(c.taddr, c.ad, c.bd, idesc(64, 128, false)); break; }
        case 18: { for (int i = 0; i < 32; i++) MMA_WS(c.taddr, c.ad, c.bd, idesc(128, 128, false)); break; }
        case 19: { for (int i = 0; i < 16; i++) { MMA_WS(c.taddr, c.ad, c.bd, idws32); MMA_WS(c.taddr + L16, c.ad, c.bd, idws32); } break; } // ws M32 alternating lanes 0/16
        case 20: { for (int i = 0; i < 8; i++) { MMA_WS(c.taddr, c.ad, c.bd, idws32); MMA_WS(c.taddr + L16, c.ad, c.bd, idws32); MMA_WS(c.taddr + (32u << 16), c.ad, c.bd, idws32); MMA_WS(c.taddr + (48u << 16), c.ad, c.bd, idws32); } break; } // ws M32 at 4 lane quads
        // ws small (M32 N256) with normal large: blocked / interleaved
        case 21: { for (int i = 0; i < 16; i++) MMA_F16(c.taddr, c.ad, c.bd, idL); for (int i = 0; i < 16; i++) MMA_WS(c.taddr + 256, c.ad, c.bd, idws32); break; }
        case 22: { for (int i = 0; i < 16; i++) { MMA_F16(c.taddr, c.ad, c.bd, idL); MMA_WS(c.taddr + 256, c.ad, c.bd, idws32); } break; }
        case 23: { for (int i = 0; i < 32; i++) MMA_WS(c.taddr + 256, c.ad, c.bd, idws32); break; }
        // mixed kinds: 16 large f16 + 16 small f8 (M128 N64)
        case 30: { for (int i = 0; i < 32; i++) MMA_F8(c.taddr + 256, c.ad, c.bd, idS8); break; }
        case 31: { for (int i = 0; i < 16; i++) MMA_F16(c.taddr, c.ad, c.bd, idL); for (int i = 0; i < 16; i++) MMA_F8(c.taddr + 256, c.ad, c.bd, idS8); break; }
        case 32: { for (int i = 0; i < 16; i++) { MMA_F16(c.taddr, c.ad, c.bd, idL); MMA_F8(c.taddr + 256, c.ad, c.bd, idS8); } break; }
        // mixed A source: small = M128 N64 f16 with A from tmem (cols 320..)
        case 40: { for (int i = 0; i < 32; i++) MMA_TS(c.taddr + 256, c.taddr + 320, c.bd, idS16); break; }
        case 41: { for (int i = 0; i < 32; i++) MMA_F16(c.taddr + 256, c.ad, c.bd, idS16); break; }
        case 42: { for (int i = 0; i < 16; i++) MMA_F16(c.taddr, c.ad, c.bd, idL); for (int i = 0; i < 16; i++) MMA_TS(c.taddr + 256, c.taddr + 320, c.bd, idS16); break; }
        case 43: { for (int i = 0; i < 16; i++) { MMA_F16(c.taddr, c.ad, c.bd, idL); MMA_TS(c.taddr + 256, c.taddr + 320, c.bd, idS16); } break; }
      }
      commit(c.mbar0); mbar_wait(c.mbar0, c.phase); t_end[0] = clk(); t_end[1] = 0;
    }
    if (lane == 0 && warp == 1 && (mode == 5 || mode == 6)) {   // two-warp lane-pair: warp1 issues M64 at lane 16 (mode 5) or lane 0 (mode 6)
      uint32_t d = c.taddr + (mode == 5 ? L16 : 0u) + 0u;
      for (int i = 0; i < 16; i++) MMA_F16(d, c.ad, c.bd, idM64);
      commit(c.mbar1); mbar_wait(c.mbar1, c.phase); t_end[1] = clk();
    }
    if (lane == 0 && warp == 0 && (mode == 5 || mode == 6)) {
      for (int i = 0; i < 16; i++) MMA_F16(c.taddr, c.ad, c.bd, idM64);
      commit(c.mbar0); mbar_wait(c.mbar0, c.phase); t_end[0] = clk();
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); __syncthreads();
    if (tid == 0) { uint64_t e = t_end[0] > t_end[1] ? t_end[0] : t_end[1]; if (r > 0) total += e - t0; }
    c.phase ^= 1; __syncthreads();
  }
  if (tid == 0) out[0] = total / (reps - 1);
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(c.taddr));
}

// ---------- two CTAs per SM: each CTA allocs 256 cols and runs `iters` x 32 MMAs of its class
__global__ void __launch_bounds__(128, 2) bench_cta(int cls_mode, int iters) {
  extern __shared__ __align__(1024) uint8_t dyn[];
  uint8_t* sA = dyn; uint8_t* sB = dyn + 128 * 32;
  __shared__ __align__(8) uint64_t mbar[2]; __shared__ uint32_t tmem_base;
  Ctx c; setup(c, sA, sB, mbar, &tmem_base, 256);
  int tid = threadIdx.x;
  int cls = (cls_mode == 0) ? 0 : (cls_mode == 1) ? 1 : (blockIdx.x & 1);   // 0 = large, 1 = small, 2 = alternate
  uint32_t idL = idesc(128, 256, false), idS = idesc(128, 32, false);
  if (tid == 0) {
    for (int it = 0; it < iters; it++) {
      if (cls == 0) { for (int i = 0; i < 32; i++) MMA_F16(c.taddr, c.ad, c.bd, idL); }
      else          { for (int i = 0; i < 32; i++) MMA_F16(c.taddr, c.ad, c.bd, idS); }
      commit(c.mbar0); mbar_wait(c.mbar0, c.phase); c.phase ^= 1;
    }
  }
  __syncthreads();
  if (tid / 32 == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" :: "r"(c.taddr));
}

static uint64_t run1(int mode, int reps = 40) { uint64_t* d; CK(cudaMalloc(&d, 8)); bench<<<1, 128>>>(d, mode, reps); CK(cudaDeviceSynchronize()); uint64_t h; CK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost)); CK(cudaFree(d)); return h; }
static float run_cta(int grid, int cls_mode, size_t smem, int iters = 200) {
  CK(cudaFuncSetAttribute(bench_cta, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  bench_cta<<<grid, 128, smem>>>(cls_mode, iters); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a)); for (int i = 0; i < 5; i++) bench_cta<<<grid, 128, smem>>>(cls_mode, iters); CK(cudaEventRecord(b)); CK(cudaDeviceSynchronize());
  float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms / 5;
}
int main(int argc, char** argv) {
  const char* what = argc > 1 ? argv[1] : "all";
  if (!strcmp(what, "all") || !strcmp(what, "lanepair")) {
    printf("[lanepair] 32 instr, N=256 K16 f16, cycles:\n");
    printf("  M128 (full datapath)                : %llu\n", (unsigned long long)run1(0));
    printf("  M64  all at lane 0                  : %llu\n", (unsigned long long)run1(1));
    printf("  M64  alternating lane 0 / lane 16   : %llu\n", (unsigned long long)run1(2));
    printf("  M64  16 at lane 0 then 16 at lane 16: %llu\n", (unsigned long long)run1(3));
    printf("  M64  alternating column halves      : %llu\n", (unsigned long long)run1(4));
    printf("  M64  two warps: lane 0 | lane 16    : %llu   (16 each)\n", (unsigned long long)run1(5));
    printf("  M64  two warps: lane 0 | lane 0     : %llu   (16 each)\n", (unsigned long long)run1(6));
  }
  if (!strcmp(what, "all") || !strcmp(what, "ws")) {
    printf("[ws] tcgen05.mma.ws kind::f16, 32 instr, cycles per instruction:\n         N=    64   128   256\n");
    int base[3] = {13, 16, 10}; const char* Ms[3] = {"32", "64", "128"};
    for (int m = 0; m < 3; m++) { printf("  M=%3s  ", Ms[m]); for (int n = 0; n < 3; n++) printf("%6.1f", run1(base[n] + m) / 32.0); printf("\n"); }
  }
  if (!strcmp(what, "all") || !strcmp(what, "wsmix")) {
    uint64_t L = run1(0), S = run1(23), B = run1(21), I = run1(22);
    printf("[wsmix] 16 x M128N256 (non-ws) + 16 x ws M32N256: large-only %llu ws-only %llu | blocked %llu interleaved %llu | additive %llu -> %.2fx / %.2fx\n",
           (unsigned long long)L, (unsigned long long)S, (unsigned long long)B, (unsigned long long)I, (unsigned long long)((L + S) / 2), B / ((L + S) / 2.0), I / ((L + S) / 2.0));
  }
  if (!strcmp(what, "all") || !strcmp(what, "kinds")) {
    uint64_t L = run1(0), S = run1(30), B = run1(31), I = run1(32);
    printf("[kinds] 16 x f16 M128N256 + 16 x f8 M128N64: large-only %llu small-only %llu | blocked %llu interleaved %llu | additive %llu -> %.2fx / %.2fx\n",
           (unsigned long long)L, (unsigned long long)S, (unsigned long long)B, (unsigned long long)I, (unsigned long long)((L + S) / 2), B / ((L + S) / 2.0), I / ((L + S) / 2.0));
  }
  if (!strcmp(what, "all") || !strcmp(what, "asrc")) {
    uint64_t L = run1(0), St = run1(40), Ss = run1(41), B = run1(42), I = run1(43);
    printf("[asrc] small M128N64 f16: A-from-tmem %llu vs A-from-smem %llu (32 instr)\n       16 large(A smem) + 16 small(A tmem): blocked %llu interleaved %llu | additive %llu -> %.2fx / %.2fx\n",
           (unsigned long long)St, (unsigned long long)Ss, (unsigned long long)B, (unsigned long long)I, (unsigned long long)((L + St) / 2), B / ((L + St) / 2.0), I / ((L + St) / 2.0));
  }
  if (!strcmp(what, "all") || !strcmp(what, "cta2")) {
    int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    size_t big = 120 * 1024, small = 12 * 1024 + 256;
    float l1 = run_cta(nsm, 0, big), s1 = run_cta(nsm, 1, big);
    float l2 = run_cta(2 * nsm, 0, small), s2 = run_cta(2 * nsm, 1, small), m2 = run_cta(2 * nsm, 2, small);
    printf("[cta2] %d SMs, per-CTA work = 200 x 32 MMAs. kernel ms:\n  1 CTA/SM: large %.3f  small(N32) %.3f\n  2 CTA/SM: large+large %.3f (%.2fx of 1/SM)  small+small %.3f (%.2fx)  large+small %.3f  vs (large + small)/1-per-SM sum %.3f -> %.2fx\n",
           nsm, l1, s1, l2, l2 / l1, s2, s2 / s1, m2, l1 + s1, m2 / (l1 + s1));
  }
  return 0;
}
