// Bech32 decode for NIP-19 nsec/npub entities.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace b32 {

inline const char* CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

inline int32_t polymod(const std::vector<int>& values) {
  const uint32_t gen[5] = {0x3b6a57b2, 0x26508e6d, 0x1ea119fa,
                           0x3d4233dd, 0x2a1462b3};
  uint32_t chk = 1;
  for (int v : values) {
    uint32_t b = chk >> 25;
    chk = ((chk & 0x1ffffff) << 5) ^ v;
    for (int i = 0; i < 5; i++)
      if ((b >> i) & 1) chk ^= gen[i];
  }
  return chk;
}

inline bool decode(const std::string& s, std::string& hrp,
                   std::vector<uint8_t>& data) {
  if (s.size() < 8) return false;
  // lowercase/uppercase consistency
  bool has_lower = false, has_upper = false;
  for (char c : s) {
    if (c >= 'a' && c <= 'z') has_lower = true;
    if (c >= 'A' && c <= 'Z') has_upper = true;
  }
  if (has_lower && has_upper) return false;
  std::string str = s;
  if (has_upper) {
    for (char& c : str) c = (char)tolower((unsigned char)c);
  }
  size_t pos = str.rfind('1');
  if (pos == std::string::npos || pos < 1 || pos + 7 > str.size()) return false;
  hrp = str.substr(0, pos);
  if (hrp.size() < 1) return false;
  for (char c : hrp) {
    if (c < 33 || c > 126) return false;
  }
  std::vector<int> values;
  for (size_t i = pos + 1; i < str.size(); i++) {
    const char* idx = strchr(CHARSET, str[i]);
    if (!idx || str[i] == 0) return false;
    values.push_back((int)(idx - CHARSET));
  }
  // checksum over hrp (expanded) + data
  std::vector<int> all;
  for (char c : hrp) all.push_back((int)((uint8_t)c) >> 5);
  all.push_back(0);
  for (char c : hrp) all.push_back((int)((uint8_t)c) & 31);
  for (int v : values) all.push_back(v);
  if (polymod(all) != 1) return false;

  // drop 6-char checksum, convert 5-bit to 8-bit (no padding allowed)
  std::vector<int> d5(values.begin(), values.end() - 6);
  int acc = 0, bits = 0;
  data.clear();
  
  for (int v : d5) {
    if (v < 0 || (v >> 5)) return false;
    acc = ((acc << 5) | v) & 0xffff;
    bits += 5;
    while (bits >= 8) {
      bits -= 8;
      data.push_back((uint8_t)((acc >> bits) & 0xff));
    }
  }
  if (bits >= 5 || ((acc << (8 - bits)) & 0xff)) return false;
  return true;
}

}  // namespace b32
