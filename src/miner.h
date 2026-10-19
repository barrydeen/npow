// Multi-GPU Nostr PoW miner interface (implementation in miner.cu).
#pragma once

#include <cstdint>
#include <vector>

// The serialized event with a 12-digit zero-padded decimal nonce placeholder.
// `tail` starts at the 64-byte boundary at or before the nonce and includes
// SHA-256 padding. The kernel patches `nonce_off..nonce_off+12` in the tail.
struct GpuJob {
  uint32_t midstate[8];
  std::vector<unsigned char> tail;
  int nonce_off = 0;  // index of first nonce digit within `tail`
  int diff = 0;
  uint64_t epoch = 0;  // bumped on every job update
};

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
  int gpuCount() const;

 private:
  void worker(size_t idx);
  struct Impl;
  Impl* impl_;
};
