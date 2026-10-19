// CUDA Nostr PoW miner.
//
// Nonce = 12 decimal digits D0..D11 (value L*1e8 + gid*100 + k):
//   D0..D3   launch index L, rendered on the host per launch
//   D4..D9   thread index gid (0..999999)
//   D10..D11 inner loop counter k (0..99)
// Each GPU takes launch indices idx, idx+ngpus, ... so devices never overlap.
//
// Per job the host precomputes the SHA-256 message schedule (+K) of every
// block after the nonce block (those blocks are constant). Per launch it
// computes the chaining state up to the block holding D4..D11 and runs the
// rounds that only touch constant words of that block. The kernel is
// specialised on R0 (first word holding a GPU-varying digit), so the
// remaining rounds are fully unrolled with compile-time word indices.
#include "miner.h"

#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include "sha256.cuh"

#define THREADS 256
#define INNER 100                    // k: two decimal digits per thread
#define GID_SPACE 1000000u           // six decimal digits per launch
#define LAUNCH_SPACE 10000u          // four decimal digits on the host
#define MIN_CHUNK (1024u * THREADS)  // threads per sub-launch, lower bound

struct LaunchParams {
  uint32_t T[16];     // nonce block words, GPU-varying digit bytes zeroed
  uint32_t st[8];     // chaining state entering the nonce block
  uint32_t v[7];      // a..g after R0 rounds (h folded into pre_t1)
  uint32_t pre_t1;    // h + S1(e) + Ch(e,f,g) + K[R0]
  uint32_t pre_t2;    // S0(a) + Maj(a,b,c)
  uint32_t sh;        // 8 * (byte offset of D4 within word R0)
  uint32_t nconst;    // constant blocks after the nonce block
  uint32_t g0, gend;  // gid range of this sub-launch
  uint32_t thr0;      // word0 threshold (fast reject)
  int diff;
  unsigned long long nonce_base;  // L * 1e8
  const uint32_t* wk;             // nconst * 64 words: W[i] + K[i]
  unsigned int* flag;
  unsigned long long* found;
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
  return __funnelshift_r(x, x, n);
}
__device__ __forceinline__ uint32_t bS0(uint32_t a) {
  return rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
}
__device__ __forceinline__ uint32_t bS1(uint32_t e) {
  return rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
}
__device__ __forceinline__ uint32_t ss0(uint32_t x) {
  return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}
__device__ __forceinline__ uint32_t ss1(uint32_t x) {
  return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}
__device__ __forceinline__ uint32_t chf(uint32_t e, uint32_t f, uint32_t g) {
  return ((f ^ g) & e) ^ g;
}
__device__ __forceinline__ uint32_t majf(uint32_t a, uint32_t b, uint32_t c) {
  return (a & b) | (c & (a | b));
}

#define ROUND(KW)                                  \
  do {                                             \
    uint32_t t1 = h + bS1(e) + chf(e, f, g) + (KW); \
    uint32_t t2 = bS0(a) + majf(a, b, c);          \
    h = g; g = f; f = e; e = d + t1;               \
    d = c; c = b; b = a; a = t1 + t2;              \
  } while (0)

template <int R0>
__global__ void __launch_bounds__(THREADS)
    npow_mine(const __grid_constant__ LaunchParams P) {
  const uint32_t gid = P.g0 + blockIdx.x * THREADS + threadIdx.x;
  if (gid >= P.gend) return;

  // D4..D9 from gid (32-bit division by constants -> mul.hi).
  uint32_t x = gid;
  const uint32_t c9 = x % 10; x /= 10;
  const uint32_t c8 = x % 10; x /= 10;
  const uint32_t c7 = x % 10; x /= 10;
  const uint32_t c6 = x % 10; x /= 10;
  const uint32_t c5 = x % 10; x /= 10;
  const uint32_t c4 = x;
  const uint32_t hi = ((c4 + 48) << 24) | ((c5 + 48) << 16) |
                      ((c6 + 48) << 8) | (c7 + 48);  // D4..D7
  const uint32_t lob = ((c8 + 48) << 24) | ((c9 + 48) << 16) | 0x3030u;

  // Word R0 only holds D4..D7 bytes: invariant over k (hoisted).
  const uint32_t w0 = P.T[R0] | (hi >> P.sh);

  uint32_t tens = 0, ones = 0;
#pragma unroll 1
  for (uint32_t k = 0; k < INNER; k++) {
    const uint32_t lo = lob | (tens << 8) | ones;  // D8..D11
    if (++ones == 10) { ones = 0; ++tens; }

    uint32_t W[16];
#pragma unroll
    for (int j = 0; j < 16; j++) W[j] = P.T[j];
    W[R0] = w0;
    W[R0 + 1] = P.T[R0 + 1] | __funnelshift_r(lo, hi, P.sh);
    if (R0 + 2 < 16) W[R0 + 2] = P.T[R0 + 2] | __funnelshift_r(0, lo, P.sh);

    // Round R0 from the host-precomputed partial sums.
    uint32_t a, b, c, d, e, f, g, h;
    {
      const uint32_t t1 = P.pre_t1 + W[R0];
      a = t1 + P.pre_t2;
      b = P.v[0]; c = P.v[1]; d = P.v[2];
      e = P.v[3] + t1;
      f = P.v[4]; g = P.v[5]; h = P.v[6];
    }
#pragma unroll
    for (int i = R0 + 1; i < 64; i++) {
      uint32_t wi;
      if (i < 16) {
        wi = W[i];
      } else {
        wi = W[i & 15] + ss0(W[(i + 1) & 15]) + W[(i + 9) & 15] +
             ss1(W[(i + 14) & 15]);
        W[i & 15] = wi;
      }
      ROUND(K256[i] + wi);
    }
    uint32_t s0 = P.st[0] + a, s1 = P.st[1] + b, s2 = P.st[2] + c,
             s3 = P.st[3] + d, s4 = P.st[4] + e, s5 = P.st[5] + f,
             s6 = P.st[6] + g, s7 = P.st[7] + h;

    // Constant trailing blocks: schedule precomputed on the host.
    for (uint32_t blk = 0; blk < P.nconst; blk++) {
      const uint32_t* wk = P.wk + blk * 64;
      a = s0; b = s1; c = s2; d = s3; e = s4; f = s5; g = s6; h = s7;
#pragma unroll
      for (int i = 0; i < 64; i++) ROUND(__ldg(wk + i));
      s0 += a; s1 += b; s2 += c; s3 += d;
      s4 += e; s5 += f; s6 += g; s7 += h;
    }

    if (s0 < P.thr0) {
      const uint32_t s[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
      int lz = 0;
      for (int i = 0; i < 8; i++) {
        if (s[i] == 0) { lz += 32; continue; }
        lz += __clz(s[i]);
        break;
      }
      if (lz >= P.diff) {
        if (atomicCAS(P.flag, 0u, 1u) == 0u)
          *P.found = P.nonce_base + (unsigned long long)gid * INNER + k;
        return;
      }
    }
  }
}

typedef void (*KernelFn)(LaunchParams);
static const KernelFn kKernels[15] = {
    npow_mine<0>,  npow_mine<1>,  npow_mine<2>,  npow_mine<3>,  npow_mine<4>,
    npow_mine<5>,  npow_mine<6>,  npow_mine<7>,  npow_mine<8>,  npow_mine<9>,
    npow_mine<10>, npow_mine<11>, npow_mine<12>, npow_mine<13>, npow_mine<14>};

// ---------- host-side SHA-256 pieces ----------

static const uint32_t hK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
static const uint32_t hIV[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                0xa54ff53a, 0x510e527f, 0x9b05688c,
                                0x1f83d9ab, 0x5be0cd19};

static inline uint32_t hror(uint32_t x, int r) {
  return (x >> r) | (x << (32 - r));
}
static inline uint32_t hS0(uint32_t a) {
  return hror(a, 2) ^ hror(a, 13) ^ hror(a, 22);
}
static inline uint32_t hS1(uint32_t e) {
  return hror(e, 6) ^ hror(e, 11) ^ hror(e, 25);
}
static inline uint32_t hCh(uint32_t e, uint32_t f, uint32_t g) {
  return (e & f) ^ (~e & g);
}
static inline uint32_t hMaj(uint32_t a, uint32_t b, uint32_t c) {
  return (a & b) ^ (a & c) ^ (b & c);
}

static void load_words(const unsigned char* p, uint32_t w[16]) {
  for (int j = 0; j < 16; j++)
    w[j] = ((uint32_t)p[j * 4] << 24) | ((uint32_t)p[j * 4 + 1] << 16) |
           ((uint32_t)p[j * 4 + 2] << 8) | (uint32_t)p[j * 4 + 3];
}

static void expand(const uint32_t w16[16], uint32_t W[64]) {
  for (int i = 0; i < 16; i++) W[i] = w16[i];
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = hror(W[i - 15], 7) ^ hror(W[i - 15], 18) ^ (W[i - 15] >> 3);
    uint32_t s1 = hror(W[i - 2], 17) ^ hror(W[i - 2], 19) ^ (W[i - 2] >> 10);
    W[i] = W[i - 16] + s0 + W[i - 7] + s1;
  }
}

// Runs rounds [0, n) on v[0..7] = a..h (n <= 16).
static void rounds(uint32_t v[8], const uint32_t w[16], int n) {
  uint32_t a = v[0], b = v[1], c = v[2], d = v[3], e = v[4], f = v[5],
           g = v[6], h = v[7];
  for (int i = 0; i < n; i++) {
    uint32_t t1 = h + hS1(e) + hCh(e, f, g) + hK[i] + w[i];
    uint32_t t2 = hS0(a) + hMaj(a, b, c);
    h = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + t2;
  }
  v[0] = a; v[1] = b; v[2] = c; v[3] = d;
  v[4] = e; v[5] = f; v[6] = g; v[7] = h;
}

static void compress(uint32_t s[8], const unsigned char* block) {
  uint32_t w16[16], W[64];
  load_words(block, w16);
  expand(w16, W);
  uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5],
           g = s[6], h = s[7];
  for (int i = 0; i < 64; i++) {
    uint32_t t1 = h + hS1(e) + hCh(e, f, g) + hK[i] + W[i];
    uint32_t t2 = hS0(a) + hMaj(a, b, c);
    h = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + t2;
  }
  s[0] += a; s[1] += b; s[2] += c; s[3] += d;
  s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}

// Per-job host state.
struct JobPrep {
  std::vector<unsigned char> msg;
  int q = 0;        // first nonce digit (D0)
  int B = 0;        // block holding D4..D11
  int R0 = 0;       // first word holding a GPU-varying digit
  uint32_t sh = 0;  // 8 * byte offset of D4 in word R0
  uint32_t nconst = 0;
  uint32_t mid[8];  // state entering block q/64
  int diff = 0;
  uint32_t thr0 = 0;
};

static uint32_t threshold_for(int diff) {
  if (diff >= 32) return 1;  // word0 must be exactly 0
  return 1u << (32 - diff);
}

static bool prep_job(const GpuJob& job, JobPrep& jp, std::vector<uint32_t>& wk,
                     std::string& err) {
  const size_t n = job.msg.size();
  const int lo = job.digit_off + 4;
  if (n == 0 || n % 64 != 0 || n > kMaxMinerMsg) {
    err = "serialized event too large or malformed (" + std::to_string(n) +
          " bytes)";
    return false;
  }
  if (job.digit_off < 0 || (size_t)job.digit_off + 12 > n ||
      lo / 64 != (lo + 7) / 64) {
    err = "nonce digits misplaced in serialized event";
    return false;
  }
  jp.msg = job.msg;
  jp.q = job.digit_off;
  jp.B = lo / 64;
  jp.R0 = (lo % 64) / 4;
  jp.sh = 8u * (uint32_t)(lo % 4);
  jp.nconst = (uint32_t)(n / 64 - jp.B - 1);
  jp.diff = job.diff;
  jp.thr0 = threshold_for(job.diff);
  // Zero the GPU-varying digit bytes so the kernel can OR them in.
  memset(jp.msg.data() + lo, 0, 8);

  memcpy(jp.mid, hIV, 32);
  for (int blk = 0; blk < jp.q / 64; blk++)
    compress(jp.mid, jp.msg.data() + blk * 64);

  wk.resize((size_t)jp.nconst * 64);
  for (uint32_t i = 0; i < jp.nconst; i++) {
    uint32_t w16[16], W[64];
    load_words(jp.msg.data() + (size_t)(jp.B + 1 + i) * 64, w16);
    expand(w16, W);
    for (int r = 0; r < 64; r++) wk[(size_t)i * 64 + r] = W[r] + hK[r];
  }
  return true;
}

// Fills the per-launch parameters for launch index L (D0..D3).
static void prep_launch(JobPrep& jp, uint32_t L, LaunchParams& P) {
  unsigned char* m = jp.msg.data();
  m[jp.q + 0] = (unsigned char)('0' + (L / 1000) % 10);
  m[jp.q + 1] = (unsigned char)('0' + (L / 100) % 10);
  m[jp.q + 2] = (unsigned char)('0' + (L / 10) % 10);
  m[jp.q + 3] = (unsigned char)('0' + L % 10);

  uint32_t st[8];
  memcpy(st, jp.mid, 32);
  for (int blk = jp.q / 64; blk < jp.B; blk++) compress(st, m + blk * 64);

  load_words(m + jp.B * 64, P.T);
  uint32_t v[8];
  memcpy(v, st, 32);
  rounds(v, P.T, jp.R0);
  memcpy(P.st, st, 32);
  for (int i = 0; i < 7; i++) P.v[i] = v[i];
  P.pre_t1 = v[7] + hS1(v[4]) + hCh(v[4], v[5], v[6]) + hK[jp.R0];
  P.pre_t2 = hS0(v[0]) + hMaj(v[0], v[1], v[2]);
  P.sh = jp.sh;
  P.nconst = jp.nconst;
  P.thr0 = jp.thr0;
  P.diff = jp.diff;
  P.nonce_base = (unsigned long long)L * GID_SPACE * INNER;
}

// ---------- Miner ----------

struct Miner::Impl {
  std::vector<int> devices;
  std::vector<std::thread> threads;
  std::atomic<bool> stop{false};
  std::atomic<bool> paused{false};
  std::atomic<uint64_t> next_epoch{1};
  std::atomic<uint64_t> job_epoch{0};
  std::atomic<bool> found{false};
  mutable std::mutex job_mtx;
  GpuJob job;
  std::mutex result_mtx;
  GpuFound result;
  std::string error;
  std::atomic<uint64_t>* counters = nullptr;

  ~Impl() {
    if (counters) delete[] counters;
  }
};

Miner::Miner(const std::vector<int>& devices) : impl_(new Impl) {
  impl_->devices = devices;
  impl_->counters = new std::atomic<uint64_t>[devices.size()];
  for (size_t i = 0; i < devices.size(); i++)
    impl_->counters[i].store(0, std::memory_order_relaxed);

  for (size_t i = 0; i < devices.size(); i++) {
    impl_->threads.emplace_back([this, i]() { worker(i); });
  }
}

Miner::~Miner() { stop(); }

void Miner::stop() {
  if (!impl_) return;
  impl_->stop.store(true);
  for (auto& t : impl_->threads)
    if (t.joinable()) t.join();
  impl_->threads.clear();
}

uint64_t Miner::setJob(const GpuJob& job) {
  uint64_t epoch;
  {
    std::lock_guard<std::mutex> lock(impl_->job_mtx);
    epoch = impl_->next_epoch.fetch_add(1);
    impl_->job = job;
    impl_->job.epoch = epoch;
    impl_->job_epoch.store(epoch);
  }
  impl_->found.store(false);  // drop any stale result from a previous job
  impl_->paused.store(false);
  return epoch;
}

void Miner::pause() { impl_->paused.store(true); }

uint64_t Miner::currentEpoch() const { return impl_->job_epoch.load(); }

bool Miner::pollFound(GpuFound& out) {
  if (!impl_->found.load()) return false;
  std::lock_guard<std::mutex> lock(impl_->result_mtx);
  out = impl_->result;
  return true;
}

double Miner::ratePerGpu(int gpu, double window_s) const {
  uint64_t h = impl_->counters[gpu].exchange(0, std::memory_order_relaxed);
  return (double)h / window_s;
}

uint64_t Miner::rawCount(int gpu) const {
  return impl_->counters[gpu].load(std::memory_order_relaxed);
}

int Miner::gpuCount() const { return (int)impl_->devices.size(); }

void Miner::worker(size_t idx) {
  const int device = impl_->devices[idx];
  const uint32_t ngpus = (uint32_t)impl_->devices.size();
  if (cudaSetDevice(device) != cudaSuccess) return;

  unsigned int* h_flag = nullptr;
  unsigned long long* h_nonce = nullptr;
  unsigned int* d_flag = nullptr;
  unsigned long long* d_nonce = nullptr;
  cudaHostAlloc(&h_flag, 4, cudaHostAllocMapped | cudaHostAllocPortable);
  cudaHostAlloc(&h_nonce, 8, cudaHostAllocMapped | cudaHostAllocPortable);
  cudaHostGetDevicePointer((void**)&d_flag, h_flag, 0);
  cudaHostGetDevicePointer((void**)&d_nonce, h_nonce, 0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  *h_flag = 0;

  uint32_t* d_wk = nullptr;
  size_t d_wk_cap = 0;  // in words

  uint64_t my_epoch = 0;
  bool job_ok = false;
  JobPrep jp;
  std::vector<uint32_t> wk;
  LaunchParams P;
  memset(&P, 0, sizeof(P));
  uint32_t L = (uint32_t)idx, g0 = 0, chunk = GID_SPACE;
  GpuJob pending_job;

  while (!impl_->stop.load()) {
    if (impl_->paused.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }
    uint64_t epoch;
    {
      std::lock_guard<std::mutex> lock(impl_->job_mtx);
      epoch = impl_->job_epoch.load();
      if (epoch != my_epoch) pending_job = impl_->job;
    }
    if (epoch != my_epoch) {
      my_epoch = epoch;
      job_ok = false;
      std::string err;
      if (!prep_job(pending_job, jp, wk, err)) {
        std::lock_guard<std::mutex> lock(impl_->result_mtx);
        impl_->error = err;
        impl_->found.store(true);
        continue;  // job_ok stays false until the next setJob
      }
      if (wk.size() > d_wk_cap) {
        if (d_wk) cudaFree(d_wk);
        d_wk_cap = wk.size();
        if (cudaMalloc(&d_wk, d_wk_cap * 4) != cudaSuccess) {
          d_wk = nullptr;
          d_wk_cap = 0;
          std::lock_guard<std::mutex> lock(impl_->result_mtx);
          impl_->error = "cudaMalloc failed";
          impl_->found.store(true);
          continue;
        }
      }
      if (!wk.empty())
        cudaMemcpy(d_wk, wk.data(), wk.size() * 4, cudaMemcpyHostToDevice);
      // Keep a sub-launch around ~1e8 block compressions, but always enough
      // threads to fill the device.
      chunk = GID_SPACE / (1 + jp.nconst);
      if (chunk < MIN_CHUNK) chunk = MIN_CHUNK;
      if (chunk > GID_SPACE) chunk = GID_SPACE;
      L = (uint32_t)idx;
      g0 = 0;
      *h_flag = 0;
      job_ok = true;
    }
    if (!job_ok) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    if (g0 == 0) prep_launch(jp, L, P);
    P.g0 = g0;
    P.gend = g0 + chunk < GID_SPACE ? g0 + chunk : GID_SPACE;
    P.wk = d_wk;
    P.flag = d_flag;
    P.found = d_nonce;
    const uint32_t nthreads = P.gend - P.g0;
    const uint32_t blocks = (nthreads + THREADS - 1) / THREADS;
    kKernels[jp.R0]<<<blocks, THREADS, 0, stream>>>(P);
    cudaStreamSynchronize(stream);
    impl_->counters[idx].fetch_add((uint64_t)nthreads * INNER,
                                   std::memory_order_relaxed);

    g0 = P.gend;
    if (g0 >= GID_SPACE) {
      g0 = 0;
      L += ngpus;
      if (L >= LAUNCH_SPACE) L = (uint32_t)idx;
    }

    if (*h_flag) {
      std::lock_guard<std::mutex> lock(impl_->result_mtx);
      impl_->result.gpu = (int)idx;
      impl_->result.nonce = *h_nonce;
      impl_->result.epoch = my_epoch;
      impl_->found.store(true);
      *h_flag = 0;
      // Keep mining until stop() — main thread verifies and decides.
    }
  }

  if (d_wk) cudaFree(d_wk);
  cudaStreamDestroy(stream);
  cudaFreeHost(h_flag);
  cudaFreeHost(h_nonce);
}
