// SHA-256 shared host/device implementation.
// The compress function operates on big-endian 32-bit words of one 64-byte block.
#pragma once

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define SHA_HD __host__ __device__ __forceinline__
#define SHA_UNROLL _Pragma("unroll")
#else
#define SHA_HD inline
#define SHA_UNROLL
#endif

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define SHA_TABLE __device__
#else
#define SHA_TABLE
#endif

SHA_TABLE const uint32_t K256[64] = {
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

SHA_TABLE const uint32_t IV256[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                  0xa54ff53a, 0x510e527f, 0x9b05688c,
                                  0x1f83d9ab, 0x5be0cd19};

SHA_HD uint32_t sha_ror(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }

// s: chaining state (host order), w_in: 16 big-endian words of the next block.
SHA_HD void sha256_block(uint32_t s[8], const uint32_t w_in[16]) {
  uint32_t w[16];
SHA_UNROLL
  for (int i = 0; i < 16; i++) w[i] = w_in[i];

  uint32_t a = s[0], b = s[1], c = s[2], d = s[3];
  uint32_t e = s[4], f = s[5], g = s[6], h = s[7];

SHA_UNROLL
  for (int i = 0; i < 64; i++) {
    uint32_t wi;
    if (i < 16) {
      wi = w[i];
    } else {
      uint32_t x = w[(i + 1) & 15];
      uint32_t s0 = sha_ror(x, 7) ^ sha_ror(x, 18) ^ (x >> 3);
      uint32_t y = w[(i + 14) & 15];
      uint32_t s1 = sha_ror(y, 17) ^ sha_ror(y, 19) ^ (y >> 10);
      wi = w[i & 15] + s0 + w[(i + 9) & 15] + s1;
      w[i & 15] = wi;
    }
    uint32_t S1 = sha_ror(e, 6) ^ sha_ror(e, 11) ^ sha_ror(e, 25);
    uint32_t ch = (e & f) ^ (~e & g);
    uint32_t t1 = h + S1 + ch + K256[i] + wi;
    uint32_t S0 = sha_ror(a, 2) ^ sha_ror(a, 13) ^ sha_ror(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = S0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  s[0] += a;
  s[1] += b;
  s[2] += c;
  s[3] += d;
  s[4] += e;
  s[5] += f;
  s[6] += g;
  s[7] += h;
}

#ifndef __CUDACC__

#include <string>

inline void sha256_host(const unsigned char* data, size_t len,
                        unsigned char out[32]) {
  uint32_t s[8];
  memcpy(s, IV256, sizeof(s));
  const size_t full = len & ~(size_t)63;
  for (size_t off = 0; off < full; off += 64) {
    uint32_t w[16];
    for (int j = 0; j < 16; j++)
      w[j] = ((uint32_t)data[off + j * 4] << 24) |
             ((uint32_t)data[off + j * 4 + 1] << 16) |
             ((uint32_t)data[off + j * 4 + 2] << 8) |
             ((uint32_t)data[off + j * 4 + 3]);
    sha256_block(s, w);
  }
  unsigned char last[128];
  size_t rem = len - full;
  memcpy(last, data + full, rem);
  last[rem] = 0x80;
  size_t padded = (rem + 9 <= 64) ? 64 : 128;
  memset(last + rem + 1, 0, padded - rem - 9);
  uint64_t bits = (uint64_t)len * 8;
  for (int i = 0; i < 8; i++) last[padded - 1 - i] = (unsigned char)(bits >> (8 * i));
  for (size_t off = 0; off < padded; off += 64) {
    uint32_t w[16];
    for (int j = 0; j < 16; j++)
      w[j] = ((uint32_t)last[off + j * 4] << 24) |
             ((uint32_t)last[off + j * 4 + 1] << 16) |
             ((uint32_t)last[off + j * 4 + 2] << 8) |
             ((uint32_t)last[off + j * 4 + 3]);
    sha256_block(s, w);
  }
  for (int i = 0; i < 8; i++) {
    out[i * 4] = (unsigned char)(s[i] >> 24);
    out[i * 4 + 1] = (unsigned char)(s[i] >> 16);
    out[i * 4 + 2] = (unsigned char)(s[i] >> 8);
    out[i * 4 + 3] = (unsigned char)s[i];
  }
}

// Midstate: state after processing nblocks full 64-byte blocks.
inline void sha256_midstate(const unsigned char* data, size_t nblocks,
                            uint32_t s_out[8]) {
  memcpy(s_out, IV256, 32);
  for (size_t off = 0; off < nblocks * 64; off += 64) {
    uint32_t w[16];
    for (int j = 0; j < 16; j++)
      w[j] = ((uint32_t)data[off + j * 4] << 24) |
             ((uint32_t)data[off + j * 4 + 1] << 16) |
             ((uint32_t)data[off + j * 4 + 2] << 8) |
             ((uint32_t)data[off + j * 4 + 3]);
    sha256_block(s_out, w);
  }
}

inline int count_lz_bits(const unsigned char id[32]) {
  int total = 0;
  for (int i = 0; i < 32; i++) {
    unsigned char b = id[i];
    if (b == 0) {
      total += 8;
      continue;
    }
    int lz = 0;
    for (int bit = 7; bit >= 0; bit--) {
      if (b & (1 << bit)) break;
      lz++;
    }
    total += lz;
    break;
  }
  return total;
}

#endif  // !__CUDACC__
