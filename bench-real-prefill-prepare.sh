#!/usr/bin/env bash
# Prepare REAL-text context chunks for bench-real-prefill.sh.
#
# Why: bench-prefill.sh feeds the engine `filler * chunks` — one sentence
# repeated. Even salted against the prefix cache, that is not a realistic
# prefill workload (tokenizer/KV behaviour on degenerate repetition differs
# from natural prose). This builds chunks of genuine book text instead.
#
# Output layout (default DATA_DIR = <script dir>/realdata):
#   src/<book>.txt                     downloaded sources (Gutenberg, PD)
#   chunks/<DEPTH>-<a|b|c>.txt         one chunk per measured run, ALL
#                                      mutually DISJOINT text ranges, so no
#                                      run can hit a prefix/context cache
#                                      warmed by any other run (any depth)
#   manifest.json                      tokens (engine-counted), sha256, source
#
# Token sizing uses the engine's own tokenizer via
# POST /v1/messages/count_tokens (strata), so "40k" means 40k engine tokens,
# not a word-count heuristic.
#
#   chunks/<DEPTH>-d1..d3.txt          DEDICATED decode-probe chunks (one per
#                                      decode run): strata's context cache
#                                      matches body SUBSEQUENCES (a leading
#                                      salt does not bust it — seen
#                                      2026-10-10: re-running the same body+
#                                      instruction hit cached=39957/39957), so
#                                      no decode run may share text with any
#                                      other run (prefill or decode)
# Env:
#   URL       Engine base URL. Default http://localhost:8082
#   DATA_DIR  Output dir. Default <script dir>/realdata
#   DEPTHS    CSV token targets. Default 40000,150000,200000
#   FORCE     1 = rebuild chunks even if manifest exists
set -euo pipefail
export PYTHONUTF8=1

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
URL="${URL:-http://localhost:8082}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/realdata}"
DEPTHS="${DEPTHS:-40000,150000,200000}"
FORCE="${FORCE:-0}"

MANIFEST="$DATA_DIR/manifest.json"
FLUSH="${FLUSH:-1}"
if [[ -f "$MANIFEST" && "$FORCE" != "1" ]]; then
  echo "manifest exists: $MANIFEST (FORCE=1 to rebuild)"
else

mkdir -p "$DATA_DIR/src" "$DATA_DIR/chunks"

# Public-domain books, plain UTF-8. Sizes give enough disjoint headroom for
# the prefill chunks (3 x (40k+150k+200k)) AND three decode chunks per depth
# (another 3 x (40k+150k+200k)) — ~2.3M tokens of unique text total.
BOOKS=(
  "2600 war_and_peace"     # ~3.4 MB, ~880k tok
  "2701 moby_dick"         # ~1.3 MB
  "345  dracula"           # ~0.9 MB
  "996  don_quixote"       # ~2.4 MB
  "1184 monte_cristo"      # ~2.3 MB
  "135  les_miserables"    # ~2.9 MB
)
for b in "${BOOKS[@]}"; do
  id=$(echo "$b" | awk '{print $1}')
  name=$(echo "$b" | awk '{print $2}')
  out="$DATA_DIR/src/$name.txt"
  if [[ -s "$out" ]]; then
    echo "have  $name ($(stat -c%s "$out") bytes)"
  else
    echo "get   $name <- gutenberg $id"
    curl -fsSL --retry 3 -m 300 -o "$out" "https://www.gutenberg.org/cache/epub/$id/pg$id.txt"
  fi
done

URL="$URL" DATA_DIR="$DATA_DIR" DEPTHS="$DEPTHS" python3 - <<'PY'
import hashlib, json, os, re, sys, time, urllib.request

URL = os.environ["URL"].rstrip("/")
DATA = os.environ["DATA_DIR"]
DEPTHS = [int(x) for x in os.environ["DEPTHS"].split(",") if x.strip()]

def count_tokens(text: str) -> int:
    body = json.dumps({
        "model": "bench-prepare",
        "messages": [{"role": "user", "content": text}],
    }).encode()
    req = urllib.request.Request(
        f"{URL}/v1/messages/count_tokens", data=body,
        headers={"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["input_tokens"]
        except Exception as e:
            if attempt == 4:
                raise
            print(f"  count_tokens retry {attempt+1}: {e}", flush=True)
            time.sleep(3)

def strip_gutenberg(t: str) -> str:
    m = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", t)
    if m:
        t = t[m.end():]
    m = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", t)
    if m:
        t = t[:m.start()]
    return t.strip()

books = {}
for fn in sorted(os.listdir(f"{DATA}/src")):
    if fn.endswith(".txt"):
        books[fn[:-4]] = strip_gutenberg(
            open(f"{DATA}/src/{fn}", encoding="utf-8", errors="replace").read())
        print(f"book {fn[:-4]}: {len(books[fn[:-4]]):,} chars")

# Disjoint allocation: each (depth, slot) gets a distinct book region.
# Cursors advance per book; a region is consumed once and never reused, so
# no chunk is a substring/prefix of any other chunk.
PLAN = {
    (40000,  "a"): "war_and_peace", (40000,  "b"): "moby_dick",   (40000,  "c"): "dracula",
    (150000, "a"): "moby_dick",     (150000, "b"): "monte_cristo",(150000, "c"): "dracula",
    (200000, "a"): "war_and_peace", (200000, "b"): "monte_cristo",(200000, "c"): "war_and_peace",
    # dedicated decode-probe chunks — one per decode run, never shared
    (40000,  "d1"): "monte_cristo",  (40000,  "d2"): "war_and_peace", (40000,  "d3"): "moby_dick",
    (150000, "d1"): "don_quixote",   (150000, "d2"): "monte_cristo",  (150000, "d3"): "les_miserables",
    (200000, "d1"): "don_quixote",   (200000, "d2"): "les_miserables",(200000, "d3"): "les_miserables",
}
# Fallback for non-default DEPTHS: round-robin books, still disjoint cursors.
ORDER = ["war_and_peace", "moby_dick", "dracula", "monte_cristo", "don_quixote", "les_miserables"]
cursor = {b: 0 for b in books}

def para_cut(t: str, pos: int) -> int:
    """Move pos forward to the next paragraph boundary."""
    nl = t.find("\n\n", pos)
    return nl + 2 if nl != -1 else len(t)

def build_chunk(book: str, target: int) -> tuple[str, int, int]:
    """Take text from cursor, sized to `target` engine tokens (+-0.5%),
    cut on paragraph boundaries. Advances the book cursor."""
    global cursor
    t = books[book]
    start = para_cut(t, cursor[book])
    if start >= len(t) - 1000:
        raise RuntimeError(f"book {book} exhausted at char {start:,}")
    # rough chars/token from a 200k-char probe
    probe_end = min(start + 200_000, len(t))
    cpt = (probe_end - start) / max(count_tokens(t[start:probe_end]), 1)
    lo = max(start, int(start + (target - target * 0.1) * cpt))
    hi = min(len(t), int(start + (target + target * 0.15) * cpt) + 10_000)
    best = None
    for _ in range(14):
        mid = (lo + hi) // 2
        end = para_cut(t, mid)
        n = count_tokens(t[start:end])
        if abs(n - target) <= target * 0.005:
            best = (end, n)
            break
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
        best = (end, n)
        if hi <= lo:
            break
    end, n = best
    cursor[book] = end
    return t[start:end], start, n

manifest = {}
for _pass in (["a", "b", "c"], ["d1", "d2", "d3"]):  # abc first (byte-identical), then decode
  for depth in DEPTHS:
    for slot in _pass:
        book = PLAN.get((depth, slot)) or ORDER[(depth // 1000 + sum(map(ord, slot))) % len(ORDER)]
        path = f"{DATA}/chunks/{depth}-{slot}.txt"
        text, cstart, ntok = build_chunk(book, depth)
        open(path, "w", encoding="utf-8").write(text)
        manifest[f"chunks/{depth}-{slot}.txt"] = {
            "target_tokens": depth,
            "tokens": ntok,
            "chars": len(text),
            "book": book,
            "char_start": cstart,
            "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        }
        print(f"{depth:>7}-{slot}  {ntok:>7,} tok  book={book:<13} chars@{cstart:,}", flush=True)

json.dump(manifest, open(f"{DATA}/manifest.json", "w"), indent=1)
print(f"\nmanifest: {DATA}/manifest.json")
PY
fi

# Flush the engine's context cache: the build's binary search fed the EXACT
# chunk texts through count_tokens, which warms strata's context memory — the
# first bench probe would then hit cached=N/N (seen 2026-10-10: 40k-d probe
# hit cached=40001 straight after prepare). One oversized unrelated request
# evicts every chunk sequence from the cache. Runs even on the skip path —
# a previous prepare may have warmed the cache.
if [[ "$FLUSH" == "1" ]]; then
URL="$URL" python3 - <<'PY2'
import json, os, random, urllib.request
URL = os.environ["URL"].rstrip("/")
_words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo " \
         "lima mike november oscar papa quebec romeo sierra tango uniform " \
         "victor whiskey xray yankee zulu".split()
_r = random.Random(99)
# size by the engine's own counter: random NATO words tokenize ~1.6-2 tok/word,
# a fixed word count can overshoot the 262144 served context (HTTP 400)
_words_txt = " ".join(_r.choice(_words) for _ in range(160_000))
def _count(t):
    _b = json.dumps({"model": "bench-prepare",
                     "messages": [{"role": "user", "content": t}]}).encode()
    _q = urllib.request.Request(f"{URL}/v1/messages/count_tokens", data=_b,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(_q, timeout=300) as _resp:
        return json.load(_resp)["input_tokens"]
_n = _count(_words_txt)
_cut = int(len(_words_txt) * min(1.0, 235_000 / max(_n, 1)))
_filler = _words_txt[:_cut]
_body = json.dumps({
    "model": "bench-prepare",
    "messages": [{"role": "user", "content": f"[flush] {_filler}"}],
    "max_tokens": 1,
}).encode()
_req = urllib.request.Request(f"{URL}/v1/chat/completions", data=_body,
                              headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(_req, timeout=1200) as _resp:
        _resp.read()
    print("engine context cache flushed (240k-token unrelated request)")
except Exception as _e:
    print(f"WARN: cache flush failed ({_e}) — first bench probe may report a cache hit")
PY2
fi
