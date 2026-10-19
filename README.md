# npow

Nostr [NIP-13](https://github.com/nostr-protocol/nips/blob/master/13.md) proof-of-work miner for notes and replies, written in CUDA. Benchmarked at **~3 GH/s per RTX 3090** (~6 GH/s on two).

Mines the event id with a midstate SHA-256 kernel (only blocks touching the nonce are recomputed), commits the target difficulty in the `nonce` tag per the NIP, refreshes `created_at` while mining, and signs with BIP-340 (libsecp256k1) once a valid id is found.

## Build

Requires CUDA toolkit, a GPU (kernels target `sm_86`), `g++-15` and libsecp256k1.

```
make            # → ./npow
./npow selftest # sha256, NIP-13 vector, BIP-340 vector, GPU round trip
```

## CLI

```
npow note -d 24 -s <nsec|hex> "content" -t '[["t","tag"]]'
npow reply --to parent.json -r wss://relay.example "reply content"
npow verify event.json        # recompute id, difficulty, committed target, sig
npow keygen
```

Replies accept the parent event JSON (file or `-` for stdin) and build NIP-10 `root`/`reply`/`p` tags automatically. `-p/--pubkey` instead of `-s/--sec` mines without signing (dry run). `--gpus 0,1` selects CUDA devices.

## HTTP backend

```
npow serve [--port 8913] [--bind 127.0.0.1] [--difficulty 24] [--gpus 0,1]
```

- `GET /` — test page (mine + sign via a NIP-07 extension)
- `GET /api/health` — `{"ok":true,"gpus":2,...}`
- `POST /api/mine` — body `{"pubkey","content","difficulty"?,"kind"?,"tags"?}` →
  `{"id","nonce","difficulty","created_at","event"}` where `event` is the unsigned
  mined event. Pass it to `window.nostr.signEvent(event)`; the extension
  recomputes the same id and attaches the signature.
- `POST /api/verify` — body is a (signed or unsigned) event →
  `{"id","difficulty","committed","target_met","sig_valid"}`

CORS is enabled. One mine request at a time (HTTP 409 if busy, 30 min cap).

## Notes

- Nonce is a 12-digit zero-padded decimal counter rendered into the last `nonce` tag; each GPU strides its own portion of the nonce space per batch.
- Difficulty > 45 is rejected by `serve` (CLI allows up to 80).
- Not audited; the server binds localhost by default and takes no secrets — signing happens client-side in the browser extension.
