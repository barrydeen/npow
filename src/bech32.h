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

// NIP-19 entity encoding (bech32, checksum constant 1).
namespace b32enc {

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

inline std::string encode(const std::string& hrp,
                          const std::vector<uint8_t>& data8) {
  std::vector<int> d5;
  int acc = 0, bits = 0;
  for (uint8_t b : data8) {
    acc = ((acc << 8) | b) & 0xfffff;
    bits += 8;
    while (bits >= 5) {
      bits -= 5;
      d5.push_back((acc >> bits) & 31);
    }
  }
  if (bits) d5.push_back((acc << (5 - bits)) & 31);
  std::vector<int> values;
  for (char c : hrp) values.push_back((int)((uint8_t)c) >> 5);
  values.push_back(0);
  for (char c : hrp) values.push_back((int)((uint8_t)c) & 31);
  values.insert(values.end(), d5.begin(), d5.end());
  values.insert(values.end(), 6, 0);
  int32_t chk = polymod(values) ^ 1;
  std::string out = hrp + '1';
  for (int v : d5) out += CHARSET[v];
  for (int i = 0; i < 6; i++) out += CHARSET[(chk >> (5 * (5 - i))) & 31];
  return out;
}

// nevent1: TLV type 0 = id, 1 = relay (repeated), 2 = author, 3 = kind (BE32).
inline std::string nevent(const std::string& id_hex,
                          const std::string& author_hex,
                          const std::vector<std::string>& relays, int kind) {
  std::vector<uint8_t> data;
  auto tlv = [&](int type, const std::vector<uint8_t>& v) {
    data.push_back((uint8_t)type);
    data.push_back((uint8_t)v.size());
    data.insert(data.end(), v.begin(), v.end());
  };
  std::vector<uint8_t> id, author;
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  for (size_t i = 0; i + 1 < id_hex.size(); i += 2)
    id.push_back((uint8_t)(nib(id_hex[i]) * 16 + nib(id_hex[i + 1])));
  for (size_t i = 0; i + 1 < author_hex.size(); i += 2)
    author.push_back((uint8_t)(nib(author_hex[i]) * 16 + nib(author_hex[i + 1])));
  tlv(0, id);
  for (auto& r : relays) {
    std::vector<uint8_t> rv(r.begin(), r.end());
    tlv(1, rv);
  }
  tlv(2, author);
  std::vector<uint8_t> kb{(uint8_t)(kind >> 24), (uint8_t)(kind >> 16),
                          (uint8_t)(kind >> 8), (uint8_t)kind};
  tlv(3, kb);
  return b32enc::encode("nevent", data);
}

}  // namespace b32enc
