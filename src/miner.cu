// CUDA Nostr PoW miner: midstate SHA-256 with per-thread nonce patching.
#include "miner.h"

#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

#include "sha256.cuh"

#define NONCE_DIGITS 12
#define UNROLL 32
#define THREADS 256
#define MAX_TAIL (48 * 1024)
#define NONCE_SPACE 1000000000000ULL  // 10^12, max 12-digit decimal

__constant__ unsigned char c_tail[MAX_TAIL];
__constant__ uint32_t c_mid[8];
__constant__ int c_tail_blocks;
__constant__ int c_var0, c_var1;
__constant__ int c_nonce_off;
__constant__ int c_diff;
__constant__ uint32_t c_thr0;

__global__ void npow_mine(uint64_t base, unsigned int* found_flag,
                          unsigned long long* found_nonce) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthreads = gridDim.x * blockDim.x;

  for (int u = 0; u < UNROLL; u++) {
    uint64_t n =
        (base + (uint64_t)u * nthreads + (uint64_t)tid) % NONCE_SPACE;
    unsigned char d[NONCE_DIGITS];
    {
      uint64_t x = n;
      for (int i = NONCE_DIGITS - 1; i >= 0; i--) {
        d[i] = (unsigned char)('0' + (x % 10));
        x /= 10;
      }
    }

    uint32_t s[8];
#pragma unroll
    for (int i = 0; i < 8; i++) s[i] = c_mid[i];

    for (int b = 0; b < c_tail_blocks; b++) {
      uint32_t w[16];
      if (b >= c_var0 && b <= c_var1) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
          uint32_t word = 0;
#pragma unroll
          for (int k = 0; k < 4; k++) {
            int idx = b * 64 + j * 4 + k;
            unsigned char ch;
            if (idx >= c_nonce_off && idx < c_nonce_off + NONCE_DIGITS)
              ch = d[idx - c_nonce_off];
            else
              ch = c_tail[idx];
            word = (word << 8) | ch;
          }
          w[j] = word;
        }
      } else {
#pragma unroll
        for (int j = 0; j < 16; j++) {
          const unsigned char* p = c_tail + b * 64 + j * 4;
          w[j] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                 ((uint32_t)p[2] << 8) | (uint32_t)p[3];
        }
      }
      sha256_block(s, w);
    }

    if (s[0] < c_thr0) {
      int lz = 0;
      for (int i = 0; i < 8; i++) {
        if (s[i] == 0) {
          lz += 32;
          continue;
        }
        lz += __clz(s[i]);
        break;
      }
      if (lz >= c_diff) {
        if (atomicCAS(found_flag, 0u, 1u) == 0u)
          *found_nonce = (unsigned long long)n;
        return;
      }
    }
  }
}

struct WorkerCtx {
  int device = 0;
  int blocks = 0;
  unsigned int* h_flag = nullptr;
  unsigned long long* h_nonce = nullptr;
  unsigned int* d_flag = nullptr;
  unsigned long long* d_nonce = nullptr;
  cudaStream_t stream = nullptr;
  uint64_t my_epoch = 0;
  uint64_t base = 0;
  bool job_ok = false;
  uint64_t batch = 0;
};

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

static uint32_t threshold_for(int diff) {
  if (diff >= 32) return 1;  // word0 must be exactly 0
  return 1u << (32 - diff);
}

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

int Miner::gpuCount() const { return (int)impl_->devices.size(); }

void Miner::worker(size_t idx) {
  WorkerCtx w;
  w.device = impl_->devices[idx];

  cudaError_t err = cudaSetDevice(w.device);
  if (err != cudaSuccess) return;

  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, w.device);
  w.blocks = sms * 16;
  if (w.blocks < 512) w.blocks = 512;

  cudaHostAlloc(&w.h_flag, 4, cudaHostAllocMapped | cudaHostAllocPortable);
  cudaHostAlloc(&w.h_nonce, 8, cudaHostAllocMapped | cudaHostAllocPortable);
  cudaHostGetDevicePointer((void**)&w.d_flag, w.h_flag, 0);
  cudaHostGetDevicePointer((void**)&w.d_nonce, w.h_nonce, 0);
  cudaStreamCreate(&w.stream);
  *w.h_flag = 0;

  w.batch = (uint64_t)w.blocks * THREADS * UNROLL;
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
      if (epoch != w.my_epoch) pending_job = impl_->job;
    }
    if (epoch != w.my_epoch) {
      GpuJob& job = pending_job;
      if (job.tail.size() > MAX_TAIL) {
        std::lock_guard<std::mutex> lock(impl_->result_mtx);
        impl_->error = "serialized event too large (" +
                       std::to_string(job.tail.size()) + " bytes tail)";
        impl_->found.store(true);
        break;
      }
      cudaMemcpyToSymbolAsync(c_tail, job.tail.data(), job.tail.size(), 0,
                              cudaMemcpyHostToDevice, w.stream);
      cudaMemcpyToSymbolAsync(c_mid, job.midstate, sizeof(job.midstate), 0,
                              cudaMemcpyHostToDevice, w.stream);
      int tail_blocks = (int)(job.tail.size() / 64);
      int var0 = job.nonce_off / 64;
      int var1 = (job.nonce_off + NONCE_DIGITS - 1) / 64;
      cudaMemcpyToSymbolAsync(c_tail_blocks, &tail_blocks, 4, 0,
                              cudaMemcpyHostToDevice, w.stream);
      cudaMemcpyToSymbolAsync(c_var0, &var0, 4, 0, cudaMemcpyHostToDevice,
                              w.stream);
      cudaMemcpyToSymbolAsync(c_var1, &var1, 4, 0, cudaMemcpyHostToDevice,
                              w.stream);
      cudaMemcpyToSymbolAsync(c_nonce_off, &job.nonce_off, 4, 0,
                              cudaMemcpyHostToDevice, w.stream);
      cudaMemcpyToSymbolAsync(c_diff, &job.diff, 4, 0, cudaMemcpyHostToDevice,
                              w.stream);
      uint32_t thr0 = threshold_for(job.diff);
      cudaMemcpyToSymbolAsync(c_thr0, &thr0, 4, 0, cudaMemcpyHostToDevice,
                              w.stream);
      cudaStreamSynchronize(w.stream);
      w.my_epoch = epoch;
      w.base = 0;
      w.job_ok = true;
      *w.h_flag = 0;
    }
    if (!w.job_ok) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    npow_mine<<<w.blocks, THREADS, 0, w.stream>>>(w.base, w.d_flag, w.d_nonce);
    cudaStreamSynchronize(w.stream);
    impl_->counters[idx].fetch_add(w.batch, std::memory_order_relaxed);
    w.base = (w.base + w.batch) % NONCE_SPACE;

    if (*w.h_flag) {
      std::lock_guard<std::mutex> lock(impl_->result_mtx);
      impl_->result.gpu = (int)idx;
      impl_->result.nonce = *w.h_nonce;
      impl_->result.epoch = w.my_epoch;
      impl_->found.store(true);
      *w.h_flag = 0;
      // Keep mining until stop() — main thread verifies and decides.
    }
  }

  cudaStreamDestroy(w.stream);
  cudaFreeHost(w.h_flag);
  cudaFreeHost(w.h_nonce);
}
