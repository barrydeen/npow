// Minimal JSON: parse (strict enough for Nostr events) + compact serialize.
// Integers are preserved exactly (created_at, kind).
#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct JVal {
  enum Type { Null, Bool, Int, Dbl, Str, Arr, Obj };
  Type t = Null;
  bool b = false;
  int64_t i = 0;
  double d = 0;
  std::string s;
  std::vector<JVal> arr;
  std::vector<std::pair<std::string, JVal>> obj;

  const JVal* find(const char* key) const {
    if (t != Obj) return nullptr;
    for (auto& kv : obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
};

namespace jdetail {

inline void jskipws(const std::string& in, size_t& p) {
  while (p < in.size() &&
         (in[p] == ' ' || in[p] == '\t' || in[p] == '\n' || in[p] == '\r'))
    p++;
}

inline void jencode_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80)
    out += (char)cp;
  else if (cp < 0x800) {
    out += (char)(0xC0 | (cp >> 6));
    out += (char)(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += (char)(0xE0 | (cp >> 12));
    out += (char)(0x80 | ((cp >> 6) & 0x3F));
    out += (char)(0x80 | (cp & 0x3F));
  } else {
    out += (char)(0xF0 | (cp >> 18));
    out += (char)(0x80 | ((cp >> 12) & 0x3F));
    out += (char)(0x80 | ((cp >> 6) & 0x3F));
    out += (char)(0x80 | (cp & 0x3F));
  }
}

inline uint32_t jhex4(const std::string& in, size_t& p) {
  if (p + 4 > in.size()) throw std::runtime_error("bad \\u escape");
  uint32_t cp = 0;
  for (int i = 0; i < 4; i++) {
    char h = in[p++];
    cp <<= 4;
    if (h >= '0' && h <= '9')
      cp |= h - '0';
    else if (h >= 'a' && h <= 'f')
      cp |= h - 'a' + 10;
    else if (h >= 'A' && h <= 'F')
      cp |= h - 'A' + 10;
    else
      throw std::runtime_error("bad \\u escape");
  }
  return cp;
}

inline std::string jparse_string(const std::string& in, size_t& p) {
  p++;  // opening quote
  std::string out;
  while (true) {
    if (p >= in.size()) throw std::runtime_error("unterminated string");
    unsigned char c = in[p];
    if (c == '"') {
      p++;
      break;
    }
    if (c == '\\') {
      p++;
      if (p >= in.size()) throw std::runtime_error("bad escape");
      char e = in[p++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = jhex4(in, p);
          if (cp >= 0xD800 && cp <= 0xDBFF && p + 1 < in.size() &&
              in[p] == '\\' && in[p + 1] == 'u') {
            size_t q = p + 2;
            if (q + 4 <= in.size()) {
              size_t save = p;
              p = q;
              uint32_t lo = jhex4(in, p);
              if (lo >= 0xDC00 && lo <= 0xDFFF)
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              else
                p = save;
            }
          }
          jencode_utf8(out, cp);
          break;
        }
        default: throw std::runtime_error("bad escape");
      }
    } else {
      out += c;
      p++;
    }
  }
  return out;
}

inline void jparse_value(const std::string& in, size_t& p, JVal& v, int depth) {
  if (depth > 64) throw std::runtime_error("nesting too deep");
  jskipws(in, p);
  if (p >= in.size()) throw std::runtime_error("unexpected end of input");
  char c = in[p];
  if (c == '{') {
    v.t = JVal::Obj;
    p++;
    jskipws(in, p);
    if (p < in.size() && in[p] == '}') {
      p++;
      return;
    }
    while (true) {
      jskipws(in, p);
      if (p >= in.size() || in[p] != '"')
        throw std::runtime_error("expected object key");
      std::string key = jparse_string(in, p);
      jskipws(in, p);
      if (p >= in.size() || in[p] != ':')
        throw std::runtime_error("expected ':'");
      p++;
      JVal val;
      jparse_value(in, p, val, depth + 1);
      v.obj.emplace_back(std::move(key), std::move(val));
      jskipws(in, p);
      if (p < in.size() && in[p] == ',') {
        p++;
        continue;
      }
      if (p < in.size() && in[p] == '}') {
        p++;
        return;
      }
      throw std::runtime_error("expected ',' or '}'");
    }
  }
  if (c == '[') {
    v.t = JVal::Arr;
    p++;
    jskipws(in, p);
    if (p < in.size() && in[p] == ']') {
      p++;
      return;
    }
    while (true) {
      JVal val;
      jparse_value(in, p, val, depth + 1);
      v.arr.push_back(std::move(val));
      jskipws(in, p);
      if (p < in.size() && in[p] == ',') {
        p++;
        continue;
      }
      if (p < in.size() && in[p] == ']') {
        p++;
        return;
      }
      throw std::runtime_error("expected ',' or ']'");
    }
  }
  if (c == '"') {
    v.t = JVal::Str;
    v.s = jparse_string(in, p);
    return;
  }
  if (in.compare(p, 4, "true") == 0) {
    v.t = JVal::Bool;
    v.b = true;
    p += 4;
    return;
  }
  if (in.compare(p, 5, "false") == 0) {
    v.t = JVal::Bool;
    v.b = false;
    p += 5;
    return;
  }
  if (in.compare(p, 4, "null") == 0) {
    v.t = JVal::Null;
    p += 4;
    return;
  }
  // number
  size_t start = p;
  if (p < in.size() && (in[p] == '-' || in[p] == '+')) p++;
  bool isint = true;
  while (p < in.size()) {
    char n = in[p];
    if (n >= '0' && n <= '9') {
      p++;
    } else if (n == '.' || n == 'e' || n == 'E' || n == '+' || n == '-') {
      isint = false;
      p++;
    } else {
      break;
    }
  }
  if (p == start) throw std::runtime_error("invalid token");
  std::string num = in.substr(start, p - start);
  if (isint) {
    v.t = JVal::Int;
    v.i = strtoll(num.c_str(), nullptr, 10);
  } else {
    v.t = JVal::Dbl;
    v.d = strtod(num.c_str(), nullptr);
  }
}

}  // namespace jdetail

inline void jparse(const std::string& in, JVal& out) {
  size_t p = 0;
  jdetail::jparse_value(in, p, out, 0);
  jdetail::jskipws(in, p);
  if (p != in.size()) throw std::runtime_error("trailing content after JSON");
}

inline std::string jescape(const std::string& raw) {
  std::string out = "\"";
  for (unsigned char c : raw) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  out += '"';
  return out;
}

inline void jdump_to(const JVal& v, std::string& out) {
  switch (v.t) {
    case JVal::Null: out += "null"; break;
    case JVal::Bool: out += v.b ? "true" : "false"; break;
    case JVal::Int: out += std::to_string(v.i); break;
    case JVal::Dbl: {
      char buf[32];
      snprintf(buf, sizeof(buf), "%.17g", v.d);
      out += buf;
      break;
    }
    case JVal::Str: out += jescape(v.s); break;
    case JVal::Arr: {
      out += '[';
      for (size_t i = 0; i < v.arr.size(); i++) {
        if (i) out += ',';
        jdump_to(v.arr[i], out);
      }
      out += ']';
      break;
    }
    case JVal::Obj: {
      out += '{';
      for (size_t i = 0; i < v.obj.size(); i++) {
        if (i) out += ',';
        out += jescape(v.obj[i].first);
        out += ':';
        jdump_to(v.obj[i].second, out);
      }
      out += '}';
      break;
    }
  }
}

inline std::string jdump(const JVal& v) {
  std::string out;
  jdump_to(v, out);
  return out;
}
