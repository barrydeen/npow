# npow

Nostr [NIP-13](https://github.com/nostr-protocol/nips/blob/master/13.md) proof-of-work miner for notes and replies, written in CUDA. Benchmarked on an RTX 3090 at **~7.8 GH/s** for short notes and ~1.8 GH/s for 280-character notes (per GPU; GPUs search disjoint nonce ranges, so rates add).

Mines the event id with a midstate SHA-256 kernel (only blocks from the nonce onward are recomputed; the rounds before the varying digits and the message schedule of the trailing constant blocks are precomputed on the host), commits the target difficulty in the `nonce` tag per the NIP, and refreshes `created_at` while mining.

**Architecture:** the backend only mines. Signing happens in the browser via a NIP-07 extension (`window.nostr.signEvent`), and publishing/fetching/relay-list lookup (NIP-65 outbox/inbox, NIP-10 reply tags, NIP-01 `q`-tag quotes) belong to the frontend with [nostr-tools](https://github.com/nbd-wtf/nostr-tools) — the event id does not commit to the signature, so signing *after* mining is valid.

## Build

Requires CUDA toolkit, a GPU (kernels target `sm_86`), `g++-15` and libsecp256k1.

```
make            # → ./npow
./npow selftest # sha256, NIP-13 vector, BIP-340 vector, GPU round trip
```

## HTTP backend

```
npow serve [--port 8913] [--bind 127.0.0.1] [--difficulty 24] [--gpus 0,1]
```

- `GET /` — test page: mine → sign via extension → publish via nostr-tools
- `GET /api/health` — `{"ok":true,"gpus":2,...}`
- `POST /api/mine` — body `{"pubkey","content","difficulty"?,"kind"?,"tags"?}` →
  `{"id","nonce","difficulty","created_at","event"}` where `event` is the unsigned
  mined event. Pass it to `window.nostr.signEvent(event)`; the extension
  recomputes the same id and attaches the signature. Assemble reply/quote tags
  in the frontend and pass them via `tags`.
- `POST /api/verify` — body is a (signed or unsigned) event →
  `{"id","difficulty","committed","target_met","sig_valid"}`

CORS is enabled. One mine request at a time (HTTP 409 if busy, 30 min cap). The server takes no secrets.

## CLI

```
npow note -d 24 -s <nsec|hex> "content" -t '[["t","tag"]]'
npow reply --to parent.json "reply content"     # NIP-10 root/reply/p tags
npow quote --to parent.json "quote content"     # q tag + p tag + nostr:nevent1… link
npow verify event.json
npow keygen
```

Parent events come from a JSON file (or `-` for stdin) — fetch them with your nostr library. Output is the signed event JSON on stdout; pipe it wherever you like. `-p/--pubkey` instead of `-s/--sec` mines without signing (dry run). `--gpus 0,1` selects CUDA devices.

## Notes

- The `nonce` tag value is a run of constant `0`s followed by a 12-digit decimal counter. The leading-zero count is chosen per event so the GPU-varying digits land in the fewest possible SHA-256 blocks, as late in their block as possible (short notes hash a single block per attempt). The tag value can therefore be up to ~75 characters long; NIP-13 only requires it to be a string.
- Digits 0–3 of the counter are a per-launch index (GPU *i* of *n* takes indices *i*, *i+n*, …), digits 4–9 are the CUDA thread, digits 10–11 an in-thread loop.
- Difficulty > 45 is rejected by `serve` (CLI allows up to 80).

## License

MIT — see [LICENSE](LICENSE).
