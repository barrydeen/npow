// npow — NIP-13 proof-of-work miner for Nostr notes and replies (CUDA).
#include "bech32.h"
#include "http.h"
#include "json.h"
#include "miner.h"
#include "sha256.cuh"

#include <cuda_runtime.h>
#include <getopt.h>
#include <secp256k1.h>
#include <secp256k1_extrakeys.h>
#include <secp256k1_schnorrsig.h>
#include <signal.h>
#include <sys/random.h>
#include <time.h>

#include <algorithm>
#include <cmath>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <mutex>
#include <thread>
#include <vector>

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop.store(true); }

static int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
static std::atomic<int64_t> g_mine_start_ns{0};
static std::atomic<bool> g_cancel{false};


static secp256k1_context* g_ctx = nullptr;

static void die(const std::string& msg) {
  fprintf(stderr, "error: %s\n", msg.c_str());
  exit(1);
}

// ---------- misc helpers ----------

static bool from_hex(const std::string& s, std::vector<unsigned char>& out) {
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  if (s.size() % 2) return false;
  out.clear();
  out.reserve(s.size() / 2);
  for (size_t i = 0; i < s.size(); i += 2) {
    int hi = nib(s[i]), lo = nib(s[i + 1]);
    if (hi < 0 || lo < 0) return false;
    out.push_back((unsigned char)(hi * 16 + lo));
  }
  return true;
}

static std::string to_hex(const unsigned char* p, size_t n) {
  static const char* hexd = "0123456789abcdef";
  std::string out;
  out.reserve(n * 2);
  for (size_t i = 0; i < n; i++) {
    out += hexd[p[i] >> 4];
    out += hexd[p[i] & 15];
  }
  return out;
}

static bool parse_key(const std::string& in, std::vector<unsigned char>& out32,
                      bool secret) {
  if (in.rfind("nsec1", 0) == 0 || in.rfind("npub1", 0) == 0) {
    std::string hrp;
    std::vector<uint8_t> data;
    if (!b32::decode(in, hrp, data)) return false;
    if (secret && hrp != "nsec") return false;
    if (!secret && hrp != "npub") return false;
    if (data.size() != 32) return false;
    out32.assign(data.begin(), data.end());
    return true;
  }
  return from_hex(in, out32) && out32.size() == 32;
}

static void rand32(unsigned char out[32]) {
  if (getrandom(out, 32, 0) != 32) die("getrandom failed");
}

static std::string pad12(uint64_t n) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%012llu",
           (unsigned long long)(n % 1000000000000ULL));
  return buf;
}

static std::string human_rate(double h) {
  char buf[40];
  if (h >= 1e9) snprintf(buf, sizeof(buf), "%.2f GH/s", h / 1e9);
  else if (h >= 1e6) snprintf(buf, sizeof(buf), "%.1f MH/s", h / 1e6);
  else if (h >= 1e3) snprintf(buf, sizeof(buf), "%.1f kH/s", h / 1e3);
  else snprintf(buf, sizeof(buf), "%.0f H/s", h);
  return buf;
}

static std::string read_all(std::istream& in) {
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open " + path);
  return read_all(f);
}

// ---------- secp256k1 ----------

static void secp_init() {
  g_ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
  if (!g_ctx) die("secp256k1 context");
}

static bool derive_xonly(const unsigned char sec[32], unsigned char pk32[32]) {
  secp256k1_pubkey pub;
  if (secp256k1_ec_pubkey_create(g_ctx, &pub, sec) != 1) return false;
  secp256k1_xonly_pubkey xonly;
  if (secp256k1_xonly_pubkey_from_pubkey(g_ctx, &xonly, nullptr, &pub) != 1)
    return false;
  return secp256k1_xonly_pubkey_serialize(g_ctx, pk32, &xonly) == 1;
}

static bool bip340_sign(const unsigned char sec[32],
                        const unsigned char msg32[32],
                        const unsigned char aux32[32],
                        unsigned char sig64[64]) {
  secp256k1_keypair kp;
  if (secp256k1_keypair_create(g_ctx, &kp, sec) != 1) return false;
  return secp256k1_schnorrsig_sign32(g_ctx, sig64, msg32, &kp, aux32) == 1;
}

static bool bip340_verify(const unsigned char pk32[32],
                          const unsigned char msg32[32],
                          const unsigned char sig64[64]) {
  secp256k1_xonly_pubkey xpk;
  if (secp256k1_xonly_pubkey_parse(g_ctx, &xpk, pk32) != 1) return false;
  return secp256k1_schnorrsig_verify(g_ctx, sig64, msg32, 32, &xpk) == 1;
}

// ---------- NIP-01 serialization ----------

struct Serialized {
  std::string ser;
  size_t nonce_off = 0;  // offset of first nonce digit in `ser`
};

static std::string raw_serialization(const std::string& pk_hex, uint64_t ts,
                                     int kind,
                                     const std::string& tags_compact,
                                     const std::string& esc_content) {
  // [0,"<pk>",<ts>,<kind>,<tags>,"<content>"]
  std::string s = "[0,\"";
  s += pk_hex;
  s += "\",";
  s += std::to_string(ts);
  s += ",";
  s += std::to_string(kind);
  s += ",";
  s += tags_compact;
  s += ",";
  s += esc_content;
  s += "]";
  return s;
}

// tags with a nonce tag appended; `inner` is the user's tags without the
// outer brackets (may be empty).
static std::string tags_with_nonce(const std::string& inner,
                                   const std::string& nonce, int diff) {
  std::string t = "[";
  if (!inner.empty()) {
    t += inner;
    t += ",";
  }
  t += "[\"nonce\",\"";
  t += nonce;
  t += "\",\"";
  t += std::to_string(diff);
  t += "\"]]";
  return t;
}

// Nonce tag value = `pad` constant '0's followed by the 12 mined digits.
static std::string nonce_value(int pad, const std::string& nonce12) {
  return std::string((size_t)pad, '0') + nonce12;
}

static Serialized build_mining_ser(const std::string& pk_hex, uint64_t ts,
                                   int kind, const std::string& inner_tags,
                                   const std::string& esc_content,
                                   const std::string& nonce12, int diff,
                                   int pad) {
  std::string A = "[0,\"" + pk_hex + "\"," + std::to_string(ts) + "," +
                  std::to_string(kind) + ",[";
  std::string B = inner_tags.empty() ? "" : inner_tags + ",";
  std::string C = "[\"nonce\",\"" + std::string((size_t)pad, '0');
  Serialized s;
  s.ser = A + B + C + nonce12 + "\",\"" + std::to_string(diff) + "\"]]," +
          esc_content + "]";
  s.nonce_off = A.size() + B.size() + C.size();
  return s;
}

// Only the SHA-256 blocks from the one holding the GPU-varying nonce digits
// (digits 4..11) onward are hashed per attempt. Leading '0's in the nonce
// value shift those digits; pick the pad that minimises hashed blocks, then
// maximises how many rounds of that block can be precomputed (digits late in
// the block), then prefers the shortest pad. The 8 varying digits must not
// straddle a block boundary.
static int choose_nonce_pad(size_t prefix_len, size_t suffix_len) {
  int best = 0;
  size_t best_blocks = SIZE_MAX, best_word = 0;
  for (int pad = 0; pad < 64; pad++) {
    size_t q = prefix_len + (size_t)pad;  // first digit
    size_t lo = q + 4;                    // first GPU-varying digit
    if (lo / 64 != (lo + 7) / 64) continue;
    size_t len = q + 12 + suffix_len;
    size_t blocks = (len + 9 + 63) / 64 - lo / 64;
    size_t word = (lo % 64) / 4;
    if (blocks < best_blocks || (blocks == best_blocks && word > best_word)) {
      best = pad;
      best_blocks = blocks;
      best_word = word;
    }
  }
  return best;
}

static int nonce_pad_for(const std::string& pk_hex, uint64_t ts, int kind,
                         const std::string& inner_tags,
                         const std::string& esc_content, int diff) {
  auto s = build_mining_ser(pk_hex, ts, kind, inner_tags, esc_content,
                            "000000000000", diff, 0);
  return choose_nonce_pad(s.nonce_off, s.ser.size() - s.nonce_off - 12);
}

static GpuJob make_job(const Serialized& s, int diff, uint64_t epoch) {
  GpuJob job;
  job.epoch = epoch;
  job.diff = diff;
  size_t len = s.ser.size();
  size_t padded = ((len + 9 + 63) / 64) * 64;
  job.msg.assign(padded, 0);
  memcpy(job.msg.data(), s.ser.data(), len);
  job.msg[len] = 0x80;
  uint64_t bits = (uint64_t)len * 8;
  for (int i = 0; i < 8; i++)
    job.msg[padded - 1 - i] = (unsigned char)(bits >> (8 * i));
  job.digit_off = (int)s.nonce_off;
  return job;
}

// ---------- NIP-10 reply tags ----------

static std::string make_reply_tags(const JVal& parent,
                                   const std::string& relay_url) {
  const JVal* pid = parent.find("id");
  const JVal* ppk = parent.find("pubkey");
  if (!pid || pid->t != JVal::Str || !ppk || ppk->t != JVal::Str)
    die("parent event missing id/pubkey");
  std::string parent_id = pid->s, parent_pk = ppk->s;
  std::string root_id = parent_id, root_relay = "";
  std::vector<std::string> ps;
  auto add_p = [&](const std::string& pk) {
    for (auto& x : ps)
      if (x == pk) return;
    ps.push_back(pk);
  };
  add_p(parent_pk);
  const JVal* ptags = parent.find("tags");
  if (ptags && ptags->t == JVal::Arr) {
    for (auto& t : ptags->arr) {
      if (t.t != JVal::Arr || t.arr.empty() || t.arr[0].t != JVal::Str)
        continue;
      const std::string& name = t.arr[0].s;
      if (name == "e" && t.arr.size() >= 2 && t.arr[1].t == JVal::Str) {
        if (t.arr.size() >= 4 && t.arr[3].t == JVal::Str &&
            t.arr[3].s == "root") {
          root_id = t.arr[1].s;
          if (t.arr.size() >= 3 && t.arr[2].t == JVal::Str &&
              root_relay.empty())
            root_relay = t.arr[2].s;
        }
      } else if (name == "p" && t.arr.size() >= 2 && t.arr[1].t == JVal::Str) {
        add_p(t.arr[1].s);
      }
    }
  }
  std::string root_relay_out =
      relay_url.empty() ? root_relay : relay_url;
  JVal out;
  out.t = JVal::Arr;
  auto add_e = [&](const std::string& id, const std::string& relay,
                   const char* marker) {
    JVal a;
    a.t = JVal::Arr;
    const char* parts[4] = {"e", id.c_str(), relay.c_str(), marker};
    for (int i = 0; i < 4; i++) {
      JVal sv;
      sv.t = JVal::Str;
      sv.s = parts[i];
      a.arr.push_back(std::move(sv));
    }
    out.arr.push_back(std::move(a));
  };
  add_e(root_id, root_relay_out, "root");
  add_e(parent_id, relay_url, "reply");
  for (auto& pk : ps) {
    JVal a;
    a.t = JVal::Arr;
    JVal s1;
    s1.t = JVal::Str;
    s1.s = "p";
    a.arr.push_back(s1);
    JVal s2;
    s2.t = JVal::Str;
    s2.s = pk;
    a.arr.push_back(std::move(s2));
    out.arr.push_back(std::move(a));
  }
  return jdump(out);
}

// ---------- mining driver ----------

struct Options {
  int diff = 24;
  std::string sec, pubkey, tags, relay, to;
  int kind = 1;
  std::vector<int> gpus;
  int64_t ts = 0;
};

static std::vector<int> parse_gpus(const std::string& spec) {
  std::vector<int> out;
  std::string part;
  for (size_t i = 0; i <= spec.size(); i++) {
    if (i == spec.size() || spec[i] == ',') {
      if (!part.empty()) {
        out.push_back(atoi(part.c_str()));
        part.clear();
      }
    } else if (isdigit((unsigned char)spec[i])) {
      part += spec[i];
    } else {
      die("bad --gpu list");
    }
  }
  if (out.empty()) die("bad --gpu list");
  return out;
}

static std::vector<int> default_gpus() {
  int n = 0;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (err != cudaSuccess || n <= 0)
    die("no CUDA devices found (is the driver loaded?)");
  std::vector<int> out;
  for (int i = 0; i < n; i++) out.push_back(i);
  return out;
}

static bool verbose_stats = false;

struct MineOutcome {
  bool ok = false;
  bool cancelled = false;
  std::string id_hex, error;
  std::string nonce_str;  // full nonce tag value (pad + 12 digits)
  uint64_t nonce = 0, ts = 0;
  double secs = 0;
};

// Shared mining core: usable by CLI (fresh Miner) and server (long-lived).
static MineOutcome mine_with(Miner& miner, const std::string& pk_hex,
                             uint64_t ts, int kind,
                             const std::string& inner_tags,
                             const std::string& esc_content, int diff,
                             double max_seconds) {
  MineOutcome out;
  GpuFound f;
  auto t0 = std::chrono::steady_clock::now();
  auto last_stats = t0;
  while (!g_stop.load() && !g_cancel.load()) {
    const int pad =
        nonce_pad_for(pk_hex, ts, kind, inner_tags, esc_content, diff);
    auto ser = build_mining_ser(pk_hex, ts, kind, inner_tags, esc_content,
                                "000000000000", diff, pad);
    GpuJob job = make_job(ser, diff, 0);
    if (job.msg.size() > kMaxMinerMsg) {
      out.error = "event too large for the miner";
      return out;
    }
    uint64_t epoch = miner.setJob(job);
    auto window_end = std::chrono::steady_clock::now() +
                      std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < window_end) {
      if (miner.pollFound(f) && f.epoch == epoch) {
        miner.pause();
        std::string nonce12 = pad12(f.nonce);
        auto fin = build_mining_ser(pk_hex, ts, kind, inner_tags, esc_content,
                                    nonce12, diff, pad);
        unsigned char id[32];
        sha256_host((const unsigned char*)fin.ser.data(), fin.ser.size(), id);
        if (count_lz_bits(id) < diff) {
          out.error = "GPU returned an id below target difficulty";
          return out;
        }
        out.ok = true;
        out.id_hex = to_hex(id, 32);
        out.nonce = f.nonce;
        out.nonce_str = nonce_value(pad, nonce12);
        out.ts = ts;
        out.secs = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
        return out;
      }
      if (g_stop.load()) break;
      if (g_cancel.load()) {
        miner.pause();
        out.cancelled = true;
        return out;
      }
      auto nowc = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(nowc - t0).count();
      if (max_seconds > 0 && elapsed > max_seconds) {
        miner.pause();
        out.error = "timed out after " + std::to_string((int)max_seconds) +
                    "s (difficulty too high)";
        return out;
      }
      if (verbose_stats && nowc - last_stats >= std::chrono::milliseconds(500)) {
        double w = std::chrono::duration<double>(nowc - last_stats).count();
        last_stats = nowc;
        fprintf(stderr, "\r  ");
        for (int i = 0; i < miner.gpuCount(); i++)
          fprintf(stderr, "gpu%d %s   ", i,
                  human_rate(miner.ratePerGpu(i, w)).c_str());
        fprintf(stderr, "| %ds", (int)elapsed);
        fflush(stderr);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    ts = (uint64_t)time(nullptr);  // refresh created_at, as NIP-13 recommends
  }
  return out;
}

// Builds the unsigned event object JSON (id, pubkey, created_at, kind, tags,
// content) — exactly what a NIP-07 extension expects from signEvent().
static std::string event_object_json(const std::string& id_hex,
                                     const std::string& pk_hex, uint64_t ts,
                                     int kind, const std::string& inner,
                                     const std::string& nonce12, int diff,
                                     const std::string& esc_content) {
  return "{\"id\":\"" + id_hex + "\",\"pubkey\":\"" + pk_hex +
         "\",\"created_at\":" + std::to_string(ts) + ",\"kind\":" +
         std::to_string(kind) + ",\"tags\":" +
          tags_with_nonce(inner, nonce12, diff) + ",\"content\":" +
          esc_content + "}";
}

enum class Mode { Note, Reply, Quote };

static int cmd_client(int argc, char** argv, Mode mode) {
  Options o;
  o.gpus = default_gpus();
  static struct option long_opts[] = {
      {"difficulty", required_argument, 0, 'd'},
      {"sec", required_argument, 0, 's'},
      {"pubkey", required_argument, 0, 'p'},
      {"tags", required_argument, 0, 't'},
      {"kind", required_argument, 0, 'k'},
      {"relay", required_argument, 0, 'r'},
      {"gpus", required_argument, 0, 'g'},
      {"ts", required_argument, 0, 1},
      {"to", required_argument, 0, 2},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};
  int c;
  while ((c = getopt_long(argc, argv, "d:s:p:t:k:r:g:h", long_opts,
                          nullptr)) != -1) {
    switch (c) {
      case 'd': o.diff = atoi(optarg); break;
      case 's': o.sec = optarg; break;
      case 'p': o.pubkey = optarg; break;
      case 't': o.tags = optarg; break;
      case 'k': o.kind = atoi(optarg); break;
      case 'r': o.relay = optarg; break;
      case 'g': o.gpus = parse_gpus(optarg); break;
      case 1: o.ts = strtoll(optarg, nullptr, 10); break;
      case 2: o.to = optarg; break;
      default: return 2;
    }
  }
  if (o.diff < 1 || o.diff > 99) die("--difficulty must be 1..99");

  // Resolve the parent event for reply/quote: --id (fetched from the big
  // public relays) or --to (local JSON file / stdin).
  JVal parent;
  std::string parent_relay_hint;
  if (mode != Mode::Note) {
    if (o.to.empty())
      die(mode == Mode::Reply ? "reply needs --to <parent-event.json|-> (fetch the parent with your nostr library or npow's web page)"
                              : "quote needs --to <parent-event.json|-> (fetch the parent with your nostr library or npow's web page)");
    std::string pjson = o.to == "-" ? read_all(std::cin) : read_file(o.to);
    jparse(pjson, parent);
  }

  unsigned char sec[32], pk32[32];
  bool have_sec = false;
  if (!o.sec.empty()) {
    std::vector<unsigned char> k;
    if (!parse_key(o.sec, k, true) || k.size() != 32)
      die("bad --sec (want 64-char hex or nsec1...)");
    memcpy(sec, k.data(), 32);
    if (secp256k1_ec_seckey_verify(g_ctx, sec) != 1)
      die("invalid secret key");
    if (!derive_xonly(sec, pk32)) die("cannot derive pubkey");
    have_sec = true;
  } else if (!o.pubkey.empty()) {
    std::vector<unsigned char> k;
    if (!parse_key(o.pubkey, k, false) || k.size() != 32)
      die("bad --pubkey (want 64-char hex or npub1...)");
    memcpy(pk32, k.data(), 32);
    fprintf(stderr,
            "note: no secret key given — id will be mined but NOT signed\n");
  } else {
    die("need --sec <hex|nsec> (to sign) or --pubkey <hex|npub> (dry run)");
  }
  std::string pk_hex = to_hex(pk32, 32);

  std::string content = optind < argc ? argv[optind] : "";
  if (content == "-") content = read_all(std::cin);
  if (content.size() > 60000) die("content too long");

  // User-provided extra tags, minus any nonce tags.
  JVal user_tags;
  user_tags.t = JVal::Arr;
  if (!o.tags.empty()) {
    JVal tv;
    jparse(o.tags, tv);
    if (tv.t != JVal::Arr) die("--tags must be a JSON array of arrays");
    for (auto& t : tv.arr) {
      bool is_nonce = t.t == JVal::Arr && !t.arr.empty() &&
                      t.arr[0].t == JVal::Str && t.arr[0].s == "nonce";
      if (!is_nonce) user_tags.arr.push_back(t);
    }
  }

  std::string inner;
  std::string esc_content = jescape(content);
  if (mode == Mode::Reply) {
    std::string eff_relay = o.relay.empty() ? parent_relay_hint : o.relay;
    std::string full = make_reply_tags(parent, eff_relay);
    inner = full.substr(1, full.size() - 2);
  } else if (mode == Mode::Quote) {
    const JVal* pid = parent.find("id");
    const JVal* ppk = parent.find("pubkey");
    const JVal* pkind = parent.find("kind");
    if (!pid || pid->t != JVal::Str || !ppk || ppk->t != JVal::Str)
      die("parent event missing id/pubkey");
    std::string eff_relay = o.relay.empty() ? parent_relay_hint : o.relay;
    std::vector<std::string> hint;
    if (!eff_relay.empty()) hint.push_back(eff_relay);
    std::string nevent = b32enc::nevent(
        pid->s, ppk->s, hint,
        pkind && pkind->t == JVal::Int ? (int)pkind->i : 1);
    esc_content = jescape(content + "\nnostr:" + nevent);
    JVal tags = user_tags;
    auto str_arr = [](std::vector<std::string> parts) {
      JVal a;
      a.t = JVal::Arr;
      for (auto& s : parts) {
        JVal sv;
        sv.t = JVal::Str;
        sv.s = s;
        a.arr.push_back(std::move(sv));
      }
      return a;
    };
    tags.arr.push_back(str_arr({"q", pid->s, eff_relay, ppk->s}));
    tags.arr.push_back(str_arr({"p", ppk->s}));
    std::string full = jdump(tags);
    inner = full.substr(1, full.size() - 2);
  } else {
    std::string full = jdump(user_tags);
    inner = full.substr(1, full.size() - 2);
  }

  uint64_t ts = o.ts ? (uint64_t)o.ts : (uint64_t)time(nullptr);
  verbose_stats = true;
  Miner miner(o.gpus);
  MineOutcome m = mine_with(miner, pk_hex, ts, o.kind, inner, esc_content,
                            o.diff, 0);
  if (!m.ok) {
    fprintf(stderr, "%s\n", m.error.empty() ? "interrupted" : m.error.c_str());
    return m.error.empty() ? 130 : 1;
  }
  uint64_t found_ts = m.ts;
  double secs = m.secs;
  unsigned char id[32];
  {
    std::vector<unsigned char> idv;
    from_hex(m.id_hex, idv);
    memcpy(id, idv.data(), 32);
  }
  std::string id_hex = m.id_hex;

  std::string sig_hex;
  if (have_sec) {
    unsigned char sig[64], aux[32];
    rand32(aux);
    if (!bip340_sign(sec, id, aux, sig)) die("schnorr signing failed");
    if (!bip340_verify(pk32, id, sig)) die("signature self-check failed");
    sig_hex = to_hex(sig, 64);
  }

  std::string out = "{\"id\":\"" + id_hex + "\",\"pubkey\":\"" + pk_hex +
                    "\",\"created_at\":" + std::to_string(found_ts) +
                    ",\"kind\":" + std::to_string(o.kind) + ",\"tags\":" +
                    tags_with_nonce(inner, m.nonce_str, o.diff) +
                    ",\"content\":" + esc_content;
  if (!sig_hex.empty()) out += ",\"sig\":\"" + sig_hex + "\"";
  out += "}";

  fprintf(stderr,
          "mined %s… diff %d in %.2fs (%s avg)\n",
          id_hex.substr(0, 16).c_str(), o.diff, secs,
          human_rate(pow(2.0, o.diff) / (secs > 0 ? secs : 1)).c_str());

  printf("%s\n", out.c_str());
  return 0;
}

// ---------- HTTP server ----------

static const char* kIndexHtml = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><title>npow — nostr PoW miner</title>
<style>
 body{background:#111;color:#ddd;font-family:ui-monospace,monospace;max-width:720px;margin:2rem auto;padding:0 1rem}
 h1{color:#7f5af0;font-size:1.4rem} h1 span{color:#666;font-weight:400;font-size:.8rem}
 label{display:block;margin:1rem 0 .3rem;color:#999;font-size:.8rem}
  input,textarea,select{width:100%;box-sizing:border-box;background:#1b1b23;color:#eee;border:1px solid #333;border-radius:6px;padding:.5rem;font-family:inherit}
  textarea{min-height:6rem} input[type=number]{width:8rem}
  button{background:#7f5af0;color:#fff;border:0;border-radius:6px;padding:.6rem 1.2rem;margin-top:1rem;cursor:pointer;font-family:inherit;font-size:.9rem}
  button:hover{filter:brightness(1.15)}
  button:disabled{opacity:.45;cursor:not-allowed}
  #noteinfo{color:#999;font-size:.75rem;margin-top:.4rem;white-space:pre-wrap;word-break:break-word}
  #hashrate{color:#7f5af0;font-size:.8rem;margin-top:.6rem;min-height:1.1rem;font-variant-numeric:tabular-nums}
  #out{background:#0c0c11;border:1px solid #333;border-radius:6px;padding:1rem;margin-top:1rem;white-space:pre-wrap;word-break:break-all;font-size:.75rem;min-height:2rem}
  #status{color:#999;font-size:.75rem}
  .row{display:flex;gap:1rem} .row>div{flex:1}
 </style></head><body>
<h1>npow <span>NIP-13 proof of work · 2× RTX 3090</span></h1>
<div id="status">checking for nostr extension…</div>
<label>pubkey (from extension)</label><input id="pk" placeholder="npub1… or hex">
<label>content</label><textarea id="content" placeholder="hello world"></textarea>
<div class="row"><div><label>difficulty (leading zero bits)</label><input id="diff" type="number" value="24" min="1" max="99"></div>
<div><label>extra tags (JSON array)</label><input id="tags" value='[["client","npow"]]'></div></div>
<div class="row"><div><label>reply / quote note (note1, nevent1 or hex)</label><input id="note" placeholder="note1… / nevent1… / 64-char hex"></div>
<div style="flex:0 0 8rem"><label>mode</label><select id="note_mode"><option value="none">none</option><option value="reply">reply</option><option value="quote">quote</option></select></div></div>
<div id="noteinfo"></div>
<label>relays to publish to (comma separated)</label>
<input id="relays" value="wss://nos.lol, wss://relay.damus.io, wss://relay.primal.net">
<button id="mine1" onclick="go(false)">mine + sign + publish</button>
<button id="mine2" onclick="go(true)" style="background:#333">mine only (no signature)</button>
<button id="cancelbtn" onclick="cancelJob()" style="background:#333;display:none">cancel</button>
<div id="hashrate"></div>
<pre id="out"></pre>
<script>
const $ = id => document.getElementById(id);
const status = t => $('status').textContent = t;
let toolsP;
const tools = () => toolsP || (toolsP = import('https://esm.sh/nostr-tools@2'));
let _pool;
const getPool = async () => _pool || (_pool = new (await tools()).SimplePool());
const parseRelays = () => $('relays').value.split(/[\s,]+/).filter(Boolean);
const setRelays = urls => $('relays').value = [...new Set(urls)].join(', ');
const fmtRate = r => r >= 1e9 ? (r/1e9).toFixed(2)+' GH/s' : r >= 1e6 ? (r/1e6).toFixed(1)+' MH/s' : (r/1e3).toFixed(0)+' kH/s';
let noteState = null;
(async () => {
  if (window.nostr && window.nostr.signEvent) {
    status('nostr extension detected (NIP-07)');
    try { $('pk').value = await window.nostr.getPublicKey(); loadOutbox(); } catch (e) {}
  } else {
    status('no NIP-07 extension detected — “mine only” still works, or paste a pubkey');
  }
})();
$('pk').addEventListener('change', () => { if ($('pk').value.trim()) loadOutbox(); });
$('note').addEventListener('change', loadNote);
$('note_mode').addEventListener('change', loadNote);
async function hexPk(v) {
  v = (v || '').trim();
  if (/^[0-9a-f]{64}$/.test(v)) return v;
  if (v.startsWith('npub1')) { const { nip19 } = await tools(); return nip19.decode(v).data; }
  return null;
}
async function relayList(bootstrap, pk, marker) {
  const p = await getPool();
  const ev = await p.get(bootstrap, {kinds:[10002], authors:[pk], limit:1});
  if (!ev) return [];
  const out = new Set();
  for (const t of ev.tags) if (t[0] === 'r' && t[1] && (!t[2] || t[2] === marker)) out.add(t[1]);
  return [...out];
}
async function loadOutbox() {
  const pk = await hexPk($('pk').value);
  if (!pk) return;
  try {
    const outbox = await relayList(parseRelays(), pk, 'write');
    if (outbox.length) { setRelays(outbox); status('outbox relays loaded from your NIP-65 list'); }
    else status('no NIP-65 relay list found — keeping default relays');
  } catch (e) { status('could not load relay list: ' + (e.message || e)); }
}
async function loadNote() {
  const raw = $('note').value.trim();
  const mode = $('note_mode').value;
  const info = $('noteinfo');
  const cleanup = () => {
    if (!noteState) return;
    const oldKeys = new Set(noteState.derived.map(t => JSON.stringify(t)));
    try { $('tags').value = JSON.stringify(JSON.parse($('tags').value || '[]').filter(t => !oldKeys.has(JSON.stringify(t)))); } catch {}
    setRelays(parseRelays().filter(u => !noteState.inbox.includes(u)));
    noteState = null;
  };
  if (!raw || mode === 'none') { cleanup(); info.textContent = ''; return; }
  const { nip19 } = await tools();
  let id = raw.toLowerCase(), authorHint = null, hintRelays = [];
  if (raw.startsWith('note1') || raw.startsWith('nevent1')) {
    let d;
    try { d = nip19.decode(raw); } catch (e) { info.textContent = 'invalid bech32 entity'; return; }
    if (d.type === 'note') id = d.data;
    else if (d.type === 'nevent') { id = d.data.id; authorHint = d.data.author || null; hintRelays = d.data.relays || []; }
    else { info.textContent = 'unsupported entity (want note1 or nevent1)'; return; }
  }
  if (!/^[0-9a-f]{64}$/.test(id)) { info.textContent = 'not a valid note id'; return; }
  info.textContent = 'fetching note…';
  const bootstrap = parseRelays();
  let ev = null;
  try { ev = await (await getPool()).get([...new Set([...bootstrap, ...hintRelays])], {ids:[id], limit:1}, {maxWait: 8000}); } catch {}
  if (!ev && !authorHint) { info.textContent = 'could not find that note on any relay'; return; }
  const author = ev ? ev.pubkey : authorHint;
  const me = await hexPk($('pk').value);
  const derived = [];
  if (mode === 'quote') {
    derived.push(['q', id, hintRelays[0] || '']);
  } else {
    let root = null;
    if (ev) {
      const etags = ev.tags.filter(t => t[0] === 'e' && t[1]);
      root = etags.find(t => t[3] === 'root') || etags[0] || null;
    }
    if (root) derived.push(['e', root[1], root[2] || '', 'root']);
    derived.push(['e', id, hintRelays[0] || '', 'reply']);
  }
  const pts = new Set([author]);
  if (mode === 'reply' && ev) for (const t of ev.tags) if (t[0] === 'p' && t[1]) pts.add(t[1]);
  pts.delete(me);
  for (const pk2 of pts) derived.push(['p', pk2]);
  let cur;
  try { cur = JSON.parse($('tags').value || '[]'); } catch { info.textContent = 'extra tags is not valid JSON — fix it and retry'; return; }
  if (noteState) {
    const oldKeys = new Set(noteState.derived.map(t => JSON.stringify(t)));
    cur = cur.filter(t => !oldKeys.has(JSON.stringify(t)));
  }
  const merged = [...cur];
  for (const t of derived) if (!merged.some(x => JSON.stringify(x) === JSON.stringify(t))) merged.push(t);
  $('tags').value = JSON.stringify(merged);
  if (noteState) setRelays(parseRelays().filter(u => !noteState.inbox.includes(u)));
  let inbox = [];
  try { inbox = await relayList(bootstrap, author, 'read'); } catch {}
  if (inbox.length) setRelays([...parseRelays(), ...inbox]);
  noteState = {id, derived, inbox};
  if (mode === 'quote') {
    const ref = 'nostr:' + nip19.neventEncode({id, relays: hintRelays.length ? hintRelays : undefined, author: author || undefined, kind: ev ? ev.kind : 1});
    if (!$('content').value.includes(ref)) $('content').value = ref + '\n\n' + $('content').value;
  }
  const short = author ? nip19.npubEncode(author).slice(0, 16) + '…' : '?';
  info.textContent = (mode === 'reply' ? 'replying to ' : 'quoting ') + short +
    (ev ? ' — “' + ev.content.slice(0, 100) + (ev.content.length > 100 ? '…' : '') + '”' : ' (note not on relays — using embedded metadata)') +
    (inbox.length ? '\n+ ' + inbox.length + ' inbox relay(s) of the author' : '');
}
async function publish(signed) {
  const urls = parseRelays();
  const { Relay } = await tools();
  let log = '\npublish:\n';
  for (const u of urls) {
    try {
      const r = await Relay.connect(u);
      await r.publish(signed);
      r.close();
      log += '  ✓ ' + u + '\n';
    } catch (e) {
      log += '  ✗ ' + u + ' — ' + (e.reason || e.message || e) + '\n';
    }
  }
  return log;
}
function setBusy(v) {
  $('mine1').disabled = v;
  $('mine2').disabled = v;
  $('cancelbtn').style.display = v ? '' : 'none';
}
async function cancelJob() { try { await fetch('/api/cancel', {method:'POST'}); } catch {} }
async function go(dry) {
  const out = $('out');
  out.textContent = 'mining…';
  $('hashrate').textContent = '';
  if (!$('pk').value) { out.textContent = 'need a pubkey (connect a NIP-07 extension or paste one)'; return; }
  let body = {content: $('content').value, difficulty: +$('diff').value, pubkey: $('pk').value.trim()};
  try { body.tags = JSON.parse($('tags').value || '[]'); } catch (e) { out.textContent = 'bad tags JSON: ' + e; return; }
  setBusy(true);
  let polling = true;
  (async () => {
    while (polling) {
      try {
        const s = await (await fetch('/api/stats')).json();
        if (s.active) $('hashrate').textContent = '⚡ ' + fmtRate(s.total) + '  (' + s.gpus.map(fmtRate).join(' + ') + ')  ·  ' + Math.floor(s.elapsed) + 's';
      } catch {}
      await new Promise(r => setTimeout(r, 500));
    }
  })();
  let r;
  try {
    r = await fetch('/api/mine', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body)});
  } catch (e) {
    polling = false; setBusy(false);
    out.textContent = 'mining failed: ' + e;
    return;
  }
  polling = false;
  setBusy(false);
  const data = await r.json();
  if (data.cancelled) { $('hashrate').textContent = ''; out.textContent = 'cancelled — lower the difficulty and try again'; return; }
  if (!r.ok) { out.textContent = 'error ' + r.status + ': ' + (data.error || '?'); return; }
  if (dry) { out.textContent = 'mined id ' + data.id + '\n\n' + JSON.stringify(data.event, null, 2); return; }
  out.textContent = 'mined id ' + data.id + '\nasking extension to sign…';
  try {
    const signed = await window.nostr.signEvent(data.event);
    const vr = await fetch('/api/verify', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(signed)});
    const v = await vr.json();
    let text = 'id:        ' + signed.id +
      '\ndifficulty: ' + v.difficulty + ' bits (committed ' + v.committed + ')' +
      '\nsignature:  ' + (v.sig_valid ? 'valid' : 'INVALID') +
      '\n\n' + JSON.stringify(signed, null, 2);
    out.textContent = text + '\npublishing via nostr-tools…';
    out.textContent = text + await publish(signed);
  } catch (e) { out.textContent = 'mined ' + data.id + ' but signing failed: ' + e; }
}
</script></body></html>)HTML";


struct VerifyResult {
  bool parsed_ok = false;
  std::string error;
  std::string id_hex;
  int lz = 0;
  int committed = -1;
  bool has_sig = false, sig_valid = false;
};

static VerifyResult verify_event(const JVal& ev) {
  VerifyResult r;
  const JVal* pk = ev.find("pubkey");
  const JVal* ts = ev.find("created_at");
  const JVal* kind = ev.find("kind");
  const JVal* tags = ev.find("tags");
  const JVal* content = ev.find("content");
  if (!pk || pk->t != JVal::Str || !ts || ts->t != JVal::Int || !kind ||
      kind->t != JVal::Int || !tags || tags->t != JVal::Arr || !content ||
      content->t != JVal::Str) {
    r.error = "event JSON missing fields";
    return r;
  }
  std::string tags_compact = jdump(*tags);
  std::string ser = raw_serialization(pk->s, (uint64_t)ts->i, (int)kind->i,
                                      tags_compact, jescape(content->s));
  unsigned char id[32];
  sha256_host((const unsigned char*)ser.data(), ser.size(), id);
  r.parsed_ok = true;
  r.id_hex = to_hex(id, 32);
  r.lz = count_lz_bits(id);
  for (auto& t : tags->arr) {
    if (t.t == JVal::Arr && t.arr.size() >= 3 && t.arr[0].t == JVal::Str &&
        t.arr[0].s == "nonce" && t.arr[2].t == JVal::Str)
      r.committed = atoi(t.arr[2].s.c_str());
  }
  if (ev.find("sig") && ev.find("sig")->t == JVal::Str) {
    r.has_sig = true;
    std::vector<unsigned char> sig, pkv;
    bool ok = from_hex(ev.find("sig")->s, sig) && sig.size() == 64 &&
              parse_key(pk->s, pkv, false) && pkv.size() == 32;
    if (ok) {
      secp256k1_xonly_pubkey xpk;
      ok = secp256k1_xonly_pubkey_parse(g_ctx, &xpk, pkv.data()) == 1;
      ok = ok && secp256k1_schnorrsig_verify(g_ctx, sig.data(), id, 32,
                                             &xpk) == 1;
    }
    r.sig_valid = ok;
  }
  return r;
}

static int cmd_verify(const std::string& path) {
  std::string json = path == "-" ? read_all(std::cin) : read_file(path);
  JVal ev;
  jparse(json, ev);
  VerifyResult v = verify_event(ev);
  if (!v.parsed_ok) die(v.error);
  printf("id:            %s\n", v.id_hex.c_str());
  if (ev.find("id") && ev.find("id")->t == JVal::Str) {
    bool match = ev.find("id")->s == v.id_hex;
    printf("claimed id:    %s (%s)\n", ev.find("id")->s.c_str(),
           match ? "MATCH" : "MISMATCH");
  }
  printf("difficulty:    %d leading zero bits\n", v.lz);
  printf("committed:     %s\n",
         v.committed >= 0 ? std::to_string(v.committed).c_str() : "(none)");
  printf("target met:    %s\n",
         (v.committed >= 0 ? (v.lz >= v.committed) : (v.lz > 0)) ? "yes" : "NO");
  if (v.has_sig) printf("signature:     %s\n", v.sig_valid ? "valid" : "INVALID");
  else printf("signature:     (none)\n");
  return 0;
}

// ---------- serve ----------

static int cmd_serve(int argc, char** argv) {
  std::string bind = "127.0.0.1";
  int port = 8913;
  int def_diff = 24;
  std::vector<int> gpus;
  bool gpus_set = false;
  static struct option long_opts[] = {
      {"port", required_argument, 0, 'p'},
      {"bind", required_argument, 0, 'b'},
      {"difficulty", required_argument, 0, 'd'},
      {"gpus", required_argument, 0, 'g'},
      {0, 0, 0, 0}};
  int c;
  while ((c = getopt_long(argc, argv, "p:b:d:g:", long_opts, nullptr)) != -1) {
    switch (c) {
      case 'p': port = atoi(optarg); break;
      case 'b': bind = optarg; break;
      case 'd': def_diff = atoi(optarg); break;
      case 'g': gpus = parse_gpus(optarg); gpus_set = true; break;
      default: return 2;
    }
  }
  if (def_diff < 1 || def_diff > 99) die("--difficulty must be 1..99");
  if (!gpus_set) gpus = default_gpus();

  static Miner miner(gpus);
  static std::mutex mine_mtx;

  auto handler = [&](const HttpRequest& req) -> HttpResponse {
    if (req.method == "OPTIONS") return {204, "text/plain", ""};
    if (req.method == "GET" &&
        (req.path == "/" || req.path == "/index.html"))
      return {200, "text/html; charset=utf-8", kIndexHtml};
    if (req.method == "GET" && req.path == "/api/health")
      return {200, "application/json",
              "{\"ok\":true,\"gpus\":" + std::to_string((int)gpus.size()) +
                  ",\"default_difficulty\":" + std::to_string(def_diff) +
                  ",\"version\":\"0.1.0\"}"};
    if (req.method == "GET" && req.path == "/api/stats") {
      int64_t start = g_mine_start_ns.load(std::memory_order_acquire);
      if (start == 0)
        return {200, "application/json", "{\"active\":false}"};
      static std::mutex stats_mtx;
      static std::vector<uint64_t> last_per_gpu;
      static int64_t last_t = 0;
      double total = 0;
      std::string arr = "[";
      {
        std::lock_guard<std::mutex> lk(stats_mtx);
        int64_t now = now_ns();
        if (last_t == 0) {
          last_t = start;
          last_per_gpu.assign(miner.gpuCount(), 0);
        }
        double dt = (now - std::max(last_t, start)) / 1e9;
        if (dt < 0.1) dt = 0.1;
        for (int i = 0; i < miner.gpuCount(); i++) {
          uint64_t cur = miner.rawCount(i);
          uint64_t prev = i < (int)last_per_gpu.size() ? last_per_gpu[i] : 0;
          double r = cur >= prev ? (double)(cur - prev) / dt : 0.0;
          if (i) arr += ",";
          arr += std::to_string((uint64_t)r);
          total += r;
          last_per_gpu[i] = cur;
        }
        last_t = now;
      }
      arr += "]";
      return {200, "application/json",
              "{\"active\":true,\"elapsed\":" +
                  std::to_string((now_ns() - start) / 1e9) +
                  ",\"total\":" + std::to_string((uint64_t)total) +
                  ",\"gpus\":" + arr + "}"};
    }
    if (req.method == "POST" && req.path == "/api/cancel") {
      g_cancel.store(true);
      return {200, "application/json", "{\"ok\":true}"};
    }
    if (req.method == "POST" && req.path == "/api/mine") {
      JVal body;
      try {
        jparse(req.body, body);
      } catch (...) {
        return {400, "application/json", "{\"error\":\"bad JSON body\"}"};
      }
      std::vector<unsigned char> pkv;
      const JVal* pk = body.find("pubkey");
      if (!pk || pk->t != JVal::Str || pk->s.empty())
        return {400, "application/json",
                "{\"error\":\"pubkey is required — call window.nostr.getPublicKey() "
                "first (or paste a 64-char hex / npub1)\"}"};
      if (!parse_key(pk->s, pkv, false) || pkv.size() != 32)
        return {400, "application/json",
                "{\"error\":\"pubkey must be 64-char hex or npub1…\"}"};
      std::string pk_hex = to_hex(pkv.data(), 32);
      const JVal* content = body.find("content");
      if (!content || content->t != JVal::Str)
        return {400, "application/json",
                "{\"error\":\"content (string) is required\"}"};
      if (content->s.size() > 60000)
        return {400, "application/json", "{\"error\":\"content too long\"}"};
      int kind = 1;
      if (body.find("kind") && body.find("kind")->t == JVal::Int)
        kind = (int)body.find("kind")->i;
      int diff = def_diff;
      if (body.find("difficulty") && body.find("difficulty")->t == JVal::Int) {
        diff = (int)body.find("difficulty")->i;
        if (diff < 1 || diff > 99)
          return {400, "application/json",
                  "{\"error\":\"difficulty must be 1..99\"}"};
      }
      std::string inner;
      if (body.find("tags") && body.find("tags")->t == JVal::Arr) {
        JVal filtered;
        filtered.t = JVal::Arr;
        for (auto& t : body.find("tags")->arr) {
          bool is_nonce = t.t == JVal::Arr && !t.arr.empty() &&
                          t.arr[0].t == JVal::Str && t.arr[0].s == "nonce";
          if (!is_nonce) filtered.arr.push_back(t);
        }
        std::string full = jdump(filtered);
        inner = full.substr(1, full.size() - 2);
      }
      uint64_t ts = (uint64_t)time(nullptr);
      if (body.find("created_at") && body.find("created_at")->t == JVal::Int)
        ts = (uint64_t)body.find("created_at")->i;

      if (!mine_mtx.try_lock())
        return {409, "application/json", "{\"error\":\"miner busy\"}"};
      g_cancel.store(false);
      g_mine_start_ns.store(now_ns(), std::memory_order_release);
      MineOutcome m =
          mine_with(miner, pk_hex, ts, kind, inner, jescape(content->s), diff,
                    1800);
      g_mine_start_ns.store(0, std::memory_order_release);
      mine_mtx.unlock();
      if (!m.ok) {
        if (m.cancelled)
          return {200, "application/json", "{\"cancelled\":true}"};
        return {500, "application/json",
                "{\"error\":\"" + m.error + "\"}"};
      }
      std::string ev = event_object_json(m.id_hex, pk_hex, m.ts, kind, inner,
                                         m.nonce_str, diff,
                                         jescape(content->s));
      std::string resp = "{\"id\":\"" + m.id_hex + "\",\"nonce\":\"" +
                         m.nonce_str + "\",\"difficulty\":" +
                         std::to_string(diff) + ",\"created_at\":" +
                         std::to_string(m.ts) + ",\"event\":" + ev + "}";
      return {200, "application/json", resp};
    }
    if (req.method == "POST" && req.path == "/api/verify") {
      JVal body;
      try {
        jparse(req.body, body);
      } catch (...) {
        return {400, "application/json", "{\"error\":\"bad JSON body\"}"};
      }
      if (body.find("event") && body.find("event")->t == JVal::Obj)
        body = *body.find("event");
      VerifyResult v = verify_event(body);
      if (!v.parsed_ok)
        return {400, "application/json", "{\"error\":\"" + v.error + "\"}"};
      std::string resp = "{\"id\":\"" + v.id_hex + "\",\"difficulty\":" +
                         std::to_string(v.lz) + ",\"committed\":" +
                         std::to_string(v.committed) + ",\"target_met\":" +
                         ((v.committed >= 0 ? v.lz >= v.committed : v.lz > 0)
                              ? "true"
                              : "false") +
                         ",\"sig_valid\":" + (v.sig_valid ? "true" : "false") +
                         "}";
      return {200, "application/json", resp};
    }
    return {404, "application/json", "{\"error\":\"not found\"}"};
  };

  int srv = http_listen(bind, port);
  if (srv < 0)
    die("cannot bind " + bind + ":" + std::to_string(port) +
        " — is another `npow serve` already running? (check: pgrep -ax npow)");
  fprintf(stderr, "npow serve: http://%s:%d (gpus:", bind.c_str(), port);
  for (int g : gpus) fprintf(stderr, " %d", g);
  fprintf(stderr, ", default difficulty %d)\n", def_diff);
  std::atomic<bool> srv_stop{false};
  http_serve(srv, srv_stop, handler);
  miner.stop();
  return 0;
}

// ---------- selftest ----------

static int cmd_selftest() {
  int fails = 0;

  // sha256 known answer
  {
    unsigned char h[32];
    sha256_host((const unsigned char*)"abc", 3, h);
    bool ok = to_hex(h, 32) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    printf("sha256(\"abc\"):        %s\n", ok ? "PASS" : "FAIL");
    if (!ok) fails++;
  }

  // NIP-13 example note
  {
    std::string tags = "[[\"nonce\",\"776797\",\"20\"]]";
    std::string ser = raw_serialization(
        "a48380f4cfcc1ad5378294fcac36439770f9c878dd880ffa94bb74ea54a6f243",
        1651794653, 1, tags,
        "\"It's just me mining my own business\"");
    unsigned char id[32];
    sha256_host((const unsigned char*)ser.data(), ser.size(), id);
    bool ok = to_hex(id, 32) ==
              "000006d8c378af1779d2feebc7603a125d99eca0ccf1085959b307f64e5dd3"
              "58";
    printf("nip13 example id:      %s\n", ok ? "PASS" : "FAIL");
    if (!ok) {
      printf("  got: %s\n", to_hex(id, 32).c_str());
      fails++;
    }
  }

  // BIP-340 test vector 0
  {
    unsigned char sk[32] = {0}, msg[32] = {0}, aux[32] = {0}, sig[64];
    sk[31] = 3;
    int ok = bip340_sign(sk, msg, aux, sig) == 1;
    if (ok) {
      ok = to_hex(sig, 64) ==
           "e907831f80848d1069a5371b402410364bdf1c5f8307b0084c55f1ce2dca8215"
           "25f66a4a85ea8b71e482a74f382d2ce5ebeee8fdb2172f477df4900d310536c0";
      unsigned char pk[32];
      ok = ok && derive_xonly(sk, pk) &&
           to_hex(pk, 32) ==
               "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9";
    }
    printf("bip340 vector 0:       %s\n", ok ? "PASS" : "FAIL");
    if (!ok) fails++;
  }

  // GPU round trip
  {
    auto gpus = default_gpus();
    unsigned char sec[32], pk[32];
    rand32(sec);
    if (!derive_xonly(sec, pk)) die("keygen");
    std::string pk_hex = to_hex(pk, 32);
    std::string esc = "\"npow gpu selftest\"";
    uint64_t ts = (uint64_t)time(nullptr);
    Miner miner(gpus);
    MineOutcome m = mine_with(miner, pk_hex, ts, 1, "", esc, 20, 60);
    uint64_t ts_out = m.ts;
    unsigned char id[32];
    {
      std::vector<unsigned char> idv;
      from_hex(m.id_hex, idv);
      memcpy(id, idv.data(), 32);
    }
    bool ok = m.ok && count_lz_bits(id) >= 20;
    // Cross-check the mined id against an independent serialization built
    // from the output-style tags (catches serialization bugs).
    if (ok) {
      std::string ser2 =
          raw_serialization(pk_hex, ts_out, 1,
                            tags_with_nonce("", m.nonce_str, 20), esc);
      unsigned char id2[32];
      sha256_host((const unsigned char*)ser2.data(), ser2.size(), id2);
      ok = memcmp(id2, id, 32) == 0;
    }
    printf("gpu mine (diff 20):    %s", ok ? "PASS" : "FAIL");
    if (ok) printf(" — %s in %.2fs", human_rate(pow(2.0, 20) / m.secs).c_str(), m.secs);
    printf("\n");
    if (!ok) fails++;
  }

  if (fails) {
    printf("SELFTEST: %d failure(s)\n", fails);
    return 1;
  }
  printf("SELFTEST: all passed\n");
  return 0;
}

// ---------- keygen ----------

static int cmd_keygen() {
  unsigned char sec[32], pk[32];
  rand32(sec);
  while (secp256k1_ec_seckey_verify(g_ctx, sec) != 1) rand32(sec);
  if (!derive_xonly(sec, pk)) die("keygen");
  printf("secret key (hex): %s\n", to_hex(sec, 32).c_str());
  printf("pubkey     (hex): %s\n", to_hex(pk, 32).c_str());
  return 0;
}

// ---------- main ----------

static void usage() {
  fprintf(stderr,
          "npow — NIP-13 proof-of-work miner for Nostr notes and replies\n"
          "\n"
          "usage:\n"
          "  npow note [opts] <content>          mine + publish a kind-1 note\n"
          "  npow reply [opts] --to <json> <content>\n"
          "                                      mine a NIP-10 reply (signing and\n"
          "                                      publishing live in the frontend)\n"
          "  npow quote [opts] --to <json> <content>\n"
          "                                      mine a NIP-01 quote note\n"
          "  npow verify <event.json|->          recompute id/difficulty/sig\n"
          "  npow serve [opts]                   HTTP backend for a web frontend\n"
          "      --port <n>                      (default 8913, binds 127.0.0.1)\n"
          "      --bind <addr>                   0.0.0.0 for LAN access\n"
          "      --difficulty <n>                default difficulty (24)\n"
          "  npow selftest                       run correctness checks\n"
          "  npow keygen                         generate a secret key\n"
          "\n"
          "options:\n"
          "  -d, --difficulty <n>   leading zero bits target (default 24)\n"
          "  -s, --sec <key>        secret key: 64-char hex or nsec1...\n"
          "  -p, --pubkey <key>     mine only, no signature: hex or npub1...\n"
          "  -t, --tags <json>      extra tags (JSON array; nonce tag added\n"
          "                         automatically, existing ones stripped)\n"
          "  -k, --kind <n>         event kind (default 1)\n"
          "  -r, --relay <url>      relay hint for e/q tags\n"
          "      --to <file|->      parent event JSON from file/stdin instead\n"
          "  -g, --gpus <list>      CUDA device ordinals (default: all)\n"
          "      --ts <unix>        fixed created_at (default: now, refreshed)\n"
          "  -h, --help\n");
}

int main(int argc, char** argv) {
  secp_init();
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  signal(SIGPIPE, SIG_IGN);

  if (argc < 2) {
    usage();
    return 2;
  }
  std::string cmd = argv[1];
  if (cmd == "-h" || cmd == "--help") {
    usage();
    return 0;
  }
  if (cmd == "note" || cmd == "reply" || cmd == "quote") {
    Mode mode = cmd == "note" ? Mode::Note
                : cmd == "reply" ? Mode::Reply : Mode::Quote;
    return cmd_client(argc - 1, argv + 1, mode);
  }
  if (cmd == "verify") {
    if (argc != 3) {
      usage();
      return 2;
    }
    return cmd_verify(argv[2]);
  }
  if (cmd == "serve") return cmd_serve(argc - 1, argv + 1);
  if (cmd == "selftest") return cmd_selftest();
  if (cmd == "keygen") return cmd_keygen();
  usage();
  return 2;
}
