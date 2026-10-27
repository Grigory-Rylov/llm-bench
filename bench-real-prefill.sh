#!/usr/bin/env bash
# Real-text prefill bench against the running engine (strata on :8082).
#
# Why this exists: bench-prefill.sh sizes its haystack as `filler * chunks`
# — one sentence repeated tens of thousands of times. The salt busts the
# prefix-cache CHAIN, but the content itself stays degenerate: engine-side
# caches/warm paths (context cache, KV reuse, tokenizer fast-paths) and even
# the attention cost profile on repeated text are not what real traffic
# looks like. This variant feeds REAL book text instead:
#
#   * context read from files: realdata/chunks/<DEPTH>-<a|b|c>.txt
#     (built by bench-real-prefill-prepare.sh, sized with the engine's own
#     tokenizer via /v1/messages/count_tokens)
#   * one chunk per measured run — run i uses chunk i, and ALL chunks across
#     ALL depths are disjoint text ranges, so no run can be served from a
#     context cache warmed by any other run
#   * a unique salt line is still prepended per request (belt-and-braces
#     against prefix-chain hits)
#   * every run reports the engine's OWN cache counters
#     (usage.prompt_tokens_details.cached_tokens and timings.cache_n):
#     a nonzero value means the run was NOT a cold prefill and is flagged
#
# Metrics per run (streaming, thinking disabled). Default view is the
# ENGINE rates (pure compute, from the engine's own timings block);
# pass CLIENT=1 to also print the client-side rates (prompt_tokens/TTFT
# and completion/(wall-TTFT) — include template, tokenization, queue):
#   TTFT            client-side, request-send -> first non-empty delta
#   engine tok/s    timings.prompt_n / timings.prompt_ms from the engine's
#                   own final-chunk timings block (pure prefill compute)
#   client tok/s    usage.prompt_tokens / TTFT (the user-truth rate, CLIENT=1)
#   decode tok/s    (probe, opt-in) completion_tokens/(wall-TTFT) client-side
#                   and timings.predicted_n/predicted_ms engine-side, measured
#                   on a continuation request at the same depth. strata
#                   IGNORES min_tokens/ignore_eos, so the length is steered
#                   by the prompt ("continue, do not stop early") and runs
#                   that stop below 90% of the target are flagged SHORT-STOP
#
# Env:
#   URL         Default http://localhost:8082
#   MODEL       Default: auto from /v1/models
#   DATA_DIR    Default <script dir>/realdata
#   DEPTHS      CSV token depths. Default 40000,150000,200000
#   RUNS        Runs per depth (each uses the next chunk file). Default 3
#   MAX_TOKENS  Completion cap. Default 16
#   DECODE_PROBE  1 = run the decode probe per depth (default 1; 0 to skip)
#   DECODE_TOKENS Completion length target for the probe. Default 128
#   DECODE_RUNS   Probe runs per depth, one dedicated chunk each (d1,d2,d3;
#                 more runs than chunks reuses them with a WARN). Default 3
#   CLIENT        1 = also show client-side rates. Default 0 (engine only)
#   QUIET         0 = verbose output (headers, chunk names, cache counters,
#                   std/CV in summaries). Default 1 = compact: per-run lines
#                   with rates + TTFT only, lean summaries (mean/min/max)
set -euo pipefail
export PYTHONUTF8=1

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
URL="${URL:-http://localhost:8082}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/realdata}"
DEPTHS="${DEPTHS:-40000,150000,200000}"
RUNS="${RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-16}"
DECODE_PROBE="${DECODE_PROBE:-1}"
DECODE_TOKENS="${DECODE_TOKENS:-128}"
DECODE_RUNS="${DECODE_RUNS:-3}"
CLIENT="${CLIENT:-0}"
QUIET="${QUIET:-1}"

[[ -d "$DATA_DIR/chunks" ]] || {
  echo "ERROR: no chunks in $DATA_DIR/chunks — run bench-real-prefill-prepare.sh first" >&2
  exit 1
}

URL="$URL" DATA_DIR="$DATA_DIR" DEPTHS="$DEPTHS" RUNS="$RUNS" MAX_TOKENS="$MAX_TOKENS" DECODE_PROBE="$DECODE_PROBE" DECODE_TOKENS="$DECODE_TOKENS" DECODE_RUNS="$DECODE_RUNS" CLIENT="$CLIENT" QUIET="$QUIET" python3 - <<'PY'
import json, os, random, statistics as s, string, sys, time
import http.client, urllib.parse
from datetime import datetime, timezone

URL = os.environ["URL"].rstrip("/")
DATA = os.environ["DATA_DIR"]
DEPTHS = [int(x) for x in os.environ["DEPTHS"].split(",") if x.strip()]
RUNS = max(1, int(os.environ["RUNS"]))
MAX_TOKENS = int(os.environ["MAX_TOKENS"])
DECODE_PROBE = os.environ.get("DECODE_PROBE", "1")
DECODE_TOKENS = max(8, int(os.environ.get("DECODE_TOKENS", "128")))
DECODE_RUNS = max(0, int(os.environ.get("DECODE_RUNS", "1")))
CLIENT = os.environ.get("CLIENT", "0") == "1"
QUIET = os.environ.get("QUIET", "1") == "1"
u = urllib.parse.urlparse(URL)
HOST, PORT = u.hostname, u.port or (443 if u.scheme == "https" else 80)
READ_TIMEOUT = 7200

def get_model():
    import urllib.request
    with urllib.request.urlopen(f"{URL}/v1/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]

MODEL = os.environ.get("MODEL") or get_model()

def salt():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + \
           "".join(random.choices(string.ascii_lowercase, k=6))

def stream_completion(prompt, max_tokens):
    """POST /v1/chat/completions with SSE streaming.
    Returns (ttft, wall, usage, timings). TTFT = first non-empty delta."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    conn = http.client.HTTPConnection(HOST, PORT, timeout=READ_TIMEOUT)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json",
                          "Content-Length": str(len(body))})
    r = conn.getresponse()
    if r.status != 200:
        msg = r.read(500).decode("utf-8", "replace")
        conn.close()
        raise RuntimeError(f"HTTP {r.status}: {msg}")
    ttft = None
    usage = timings = None
    while True:
        line = r.readline()
        if not line:
            break
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if ev.get("usage"):
            usage = ev["usage"]
        if ev.get("timings"):
            timings = ev["timings"]
        for ch in ev.get("choices", []):
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
    wall = time.perf_counter() - t0
    conn.close()
    if ttft is None:
        raise RuntimeError("no token received")
    if usage is None:
        raise RuntimeError("no usage in stream (engine did not echo prompt_tokens)")
    return ttft, wall, usage, timings

def warmup(depth):
    # DEPTH-SIZED throwaway: the first BIG request after idle is an outlier
    # on strata (seen: 67s TTFT on the first 40k run, 4x cold) — a 2k warmup
    # does not absorb it. Filler is unrelated text, never a chunk.
    filler = ("warmup pass for the real-text prefill bench, plain on purpose. "
              * (depth // 12 + 1))
    stream_completion(f"[warm {salt()}] {filler}", 4)

manifest = {}
try:
    manifest = json.load(open(f"{DATA}/manifest.json"))
except FileNotFoundError:
    pass

def stats_compact(name, vals, unit=""):
    # compact: mean only (+min/max where the caller asks), no std/CV
    m = s.mean(vals)
    return f"{name:<13} mean={m:8.1f}{unit}"

def stats(name, vals):
    m = s.mean(vals)
    sd = s.stdev(vals) if len(vals) > 1 else 0.0
    cv = sd / m * 100 if m else 0
    return f"  {name:<14} mean={m:8.1f}  std={sd:7.1f}  CV={cv:5.2f}%"

if not QUIET:
    print(f"model   {MODEL}")
    print(f"url     {URL}")
    print(f"data    {DATA}/chunks")
    print(f"depths  {DEPTHS}  runs/depth={RUNS}  max_tokens={MAX_TOKENS}  "
          f"thinking=off  rates={'engine+client' if CLIENT else 'engine (CLIENT=1 for both)'}")

for depth in DEPTHS:
    slots = "abc"
    files = [(f"{depth}-{sl}.txt", f"{DATA}/chunks/{depth}-{sl}.txt")
             for sl in slots if os.path.exists(f"{DATA}/chunks/{depth}-{sl}.txt")]
    if not files:
        print(f"\n========== real-prefill-{depth//1000}k ==========\\n  SKIP: no chunks for depth {depth} — run prepare")
        continue
    label = f"real-prefill-{depth//1000}k"
    if not QUIET:
        print(f"\n========== {label.upper()} (depth={depth}, {len(files)} chunk(s), one per run) ==========")
    warmup(depth)
    if DECODE_PROBE == "1" and DECODE_RUNS > 0 and files:
        if not QUIET:
            print(f"\n=== decode probe [{label}] (continuation, target={DECODE_TOKENS} tokens) ===")
        d_client, d_engine, d_acc = [], [], []
        for i in range(DECODE_RUNS):
            # one dedicated decode chunk per decode run (d1, d2, d3): strata's
            # context cache matches body subsequences (salt does NOT bust it),
            # so no decode run may share text with any other run — see prepare
            dpath = f"{DATA}/chunks/{depth}-d{i + 1}.txt"
            if os.path.exists(dpath):
                name, path = f"{depth}-d{i + 1}.txt", dpath
            else:
                name, path = files[i % len(files)]
                if not QUIET:
                    print(f"  dec-{i+1}  WARN: no {depth}-d{i+1}.txt (re-run prepare) — "
                          f"sharing {name} with other runs; cache hit likely")
            text = open(path, encoding="utf-8").read()
            prompt = (f"[bench {salt()}] Read the following text, then CONTINUE it in the "
                      f"same style and voice, picking up exactly where it stops. Write at "
                      f"least {int(DECODE_TOKENS * 1.5)} words. Do not stop early, do not "
                      f"summarize, do not comment.\n\n" + text)
            try:
                ttft, wall, usage, tm = stream_completion(prompt, DECODE_TOKENS)
            except Exception as e:
                print(f"  dec-{i+1}   FAIL: {e}")
                continue
            comp = usage.get("completion_tokens", 0)
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            cache_n = (tm or {}).get("cache_n", 0)
            dw = wall - ttft
            dtps = comp / dw if dw > 0.05 else 0.0
            e_dtps = 0.0
            if tm and tm.get("predicted_n") and tm.get("predicted_ms"):
                e_dtps = tm["predicted_n"] / (tm["predicted_ms"] / 1000.0)
            dn = (tm or {}).get("draft_n", 0)
            da = (tm or {}).get("draft_n_accepted", 0)
            acc = f"  MTP acc={da}/{dn} ({da/dn*100:.0f}%)" if (dn and not QUIET) else ""
            short = "  ⚠ SHORT-STOP" if (comp < DECODE_TOKENS * 0.9 and not QUIET) else ""
            if CLIENT or not e_dtps:
                drate = (f"decode={e_dtps:6.1f} (engine)  {dtps:6.1f} (client) tok/s"
                         if e_dtps else
                         f"decode={dtps:6.1f} tok/s (client; engine timings missing)")
            else:
                drate = f"decode={e_dtps:6.1f} tok/s (engine)"
            if QUIET:
                # compact: attempts + speed + ttft only
                qrate = e_dtps if e_dtps else dtps
                qtag = "" if e_dtps else " (client)"
                print(f"  dec-{i+1}  decode={qrate:6.1f}{qtag} tok/s  ttft={ttft*1000:7.0f}ms{acc}{short}")
            else:
                print(f"  dec-{i+1}  {name:<15} comp={comp:>4}  {drate}  "
                      f"ttft={ttft*1000:7.0f}ms{acc}  cached={cached}/{cache_n}{short}")
            if dtps:
                d_client.append(dtps)
            if e_dtps:
                d_engine.append(e_dtps)
            if dn:
                d_acc.append(da / dn * 100)
        if d_client or d_engine:
            if QUIET:
                vals = d_engine if d_engine else d_client
                print(f"  decode tok/s   mean={s.mean(vals):8.1f}")
                if d_acc:
                    print(f"  MTP accept     mean={s.mean(d_acc):5.1f}%")
            else:
                print(f"\n=== summary [{label}-decode] "
                      f"(n={max(len(d_client), len(d_engine))}) ===")
                if CLIENT:
                    print(stats("decode tok/s", d_client) + "   (completion/(wall-TTFT))")
                if d_engine:
                    print(stats("decode tok/s", d_engine) + "   (timings predicted_n/predicted_ms)")
                elif not CLIENT:
                    print(stats("decode tok/s", d_client) + "   (fallback: engine timings missing)")
                if d_acc:
                    print(f"  MTP accept     mean={s.mean(d_acc):5.1f}%")
    if not QUIET:
        print("=== prefill measured ===")
    c_pps, e_pps, ttfts, ptoks, cached_max = [], [], [], [], 0
    for i in range(RUNS):
        name, path = files[i % len(files)]
        if i >= len(files) and not QUIET:
            print(f"  run-{i+1}  WARN: reusing chunk {name} (run {i+1} > {len(files)} chunks) —"
                  f" context cache may serve it; treat as contaminated")
        text = open(path, encoding="utf-8").read()
        prompt = f"[bench {salt()}] Read the following text. Reply with the single word DONE.\n\n{text}"
        try:
            ttft, wall, usage, tm = stream_completion(prompt, MAX_TOKENS)
        except Exception as e:
            print(f"  run-{i+1}  FAIL: {e}")
            continue
        p = usage.get("prompt_tokens", 0)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        cache_n = (tm or {}).get("cache_n", 0)
        cached_max = max(cached_max, cached, cache_n or 0)
        cpps = p / ttft if ttft > 0 else 0
        # engine-side pure prefill rate: recompute from prompt_n/prompt_ms
        # rather than trusting prompt_per_second's unit
        eps = 0.0
        if tm and tm.get("prompt_n") and tm.get("prompt_ms"):
            eps = tm["prompt_n"] / (tm["prompt_ms"] / 1000.0)
        flag = "  ⚠ CACHE HIT" if (cached or cache_n) else ""
        if QUIET:
            # compact: attempts + speed + ttft only
            qrate = eps if eps else cpps
            qtag = "" if eps else " (client)"
            warn = "  ⚠ CACHE" if (cached or cache_n) else ""
            print(f"  run-{i+1}  prefill={qrate:7.1f}{qtag} tok/s  ttft={ttft*1000:8.0f}ms{warn}")
        else:
            if CLIENT or not eps:
                rate = (f"engine={eps:7.1f}  client={cpps:7.1f} tok/s" if eps
                        else f"client={cpps:7.1f} tok/s (engine timings missing)")
            else:
                rate = f"engine={eps:7.1f} tok/s"
            print(f"  run-{i+1}  {name:<15} ptok={p:>7,}  ttft={ttft*1000:8.0f}ms  "
                  f"{rate}  cached={cached}/{cache_n}{flag}")
        c_pps.append(cpps); ttfts.append(ttft); ptoks.append(p)
        if eps:
            e_pps.append(eps)
    if c_pps:
        if QUIET:
            pvals = e_pps if e_pps else c_pps
            print(f"  prefill tok/s  mean={s.mean(pvals):8.1f}")
            print(f"  TTFT           mean={s.mean(ttfts)*1000:8.0f}ms  "
                  f"min={min(ttfts)*1000:.0f}ms  max={max(ttfts)*1000:.0f}ms")
            if cached_max:
                print(f"  ⚠ cache reuse up to {cached_max:,} tokens")
        else:
            print(f"\n=== summary [{label}] (n={len(c_pps)}) ===")
            if CLIENT:
                print(stats("client tok/s", c_pps) + "   (prompt_tokens/TTFT)")
            if e_pps:
                print(stats("engine tok/s", e_pps) + "   (timings prompt_n/prompt_ms)")
            elif not CLIENT:
                print(stats("client tok/s", c_pps) + "   (fallback: engine timings missing)")
            print(f"  TTFT           mean={s.mean(ttfts)*1000:8.0f}ms  "
                  f"min={min(ttfts)*1000:.0f}ms  max={max(ttfts)*1000:.0f}ms")
            print(f"  prompt tokens  mean={s.mean(ptoks):,.0f}")
            verdict = "clean (no cache reuse)" if cached_max == 0 else \
                      f"⚠ CONTAMINATED: up to {cached_max:,} tokens served from cache"
            print(f"  cache reuse    {verdict}")
PY
