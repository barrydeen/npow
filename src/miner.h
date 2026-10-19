// Multi-GPU Nostr PoW miner interface (implementation in miner.cu).
#pragma once

#include <cstdint>
#include <vector>

// The full serialized event, SHA-256 padded (length is a multiple of 64),
// with a 12-digit decimal nonce placeholder at `digit_off`. The miner owns
// the nonce space: digits 0-3 are set per launch (host), digits 4-11 vary on
// the GPU. The 8 GPU-varying digits (digit_off+4 .. +11) must not straddle a
// 64-byte block boundary.
struct GpuJob {
  std::vector<unsigned char> msg;
  int digit_off = 0;  // index of the first of the 12 nonce digits in `msg`
  int diff = 0;
  uint64_t epoch = 0;  // bumped on every job update
};

// Largest padded message the miner accepts.
constexpr size_t kMaxMinerMsg = 64 * 1024;

struct GpuFound {
  int gpu = -1;
  uint64_t nonce = 0;
  uint64_t epoch = 0;
};

class Miner {
 public:
  explicit Miner(const std::vector<int>& devices);
  ~Miner();

  // Publish a new job; workers upload it to their device and reset counters.
  // Also resumes mining (clears pause()). Returns the job's epoch, which is
  // assigned by the miner (monotonic across its lifetime).
  uint64_t setJob(const GpuJob& job);

  // Stop mining the current job (workers idle until the next setJob).
  void pause();

  // Epoch of the most recently published job (for matching pollFound results).
  uint64_t currentEpoch() const;

  // Non-blocking. Returns true once per successful find (per job epoch).
  bool pollFound(GpuFound& out);

  void stop();

  // Aggregate hashes/sec over the last `window_ms`; resets counters.
  double ratePerGpu(int gpu, double window_s) const;

  // Non-destructive read of the per-GPU hash counter (for live stats polling).
  uint64_t rawCount(int gpu) const;

  int gpuCount() const;

 private:
  void worker(size_t idx);
  struct Impl;
  Impl* impl_;
};
