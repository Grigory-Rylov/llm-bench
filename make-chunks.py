#!/usr/bin/env python3
"""Строит ДОПОЛНИТЕЛЬНЫЕ непересекающиеся чанки для bench-parallel-decode.py.

Зачем: prepare делает на глубину 6 слотов (-a -b -c -d1 -d2 -d3). Бенчу на N
потоков нужно N×runs своих текстов, и ни один не должен совпадать с текстом
другого прогона: контекст-кэш strata матчит ТЕЛО запроса (salt в префиксе его
не бьёт), поэтому повторно поданный чанк обслуживается из кэша, и его цифры
получаются выше, чем у конкурентов.

Чанки пишутся числовыми слотами <DEPTH>-<SLOT>.txt рядом с буквенными чанками
prepare; курсоры по книгам стартуют за край всего уже использованного — по
файлам на диске, а не только по manifest.json (прошлая сборка могла умереть до
его записи, и тогда её тексты остались бы на диске неучтёнными: следующий прогон
дал бы по ним cache hit). Поэтому новые тексты не пересекаются ни между собой,
ни с чанками prepare. Слоты, размеры, книги и sha — в тот же manifest, на каждый
чанк, а не один раз в конце.

  URL=http://localhost:8082 ./make-chunks.py --depth 40000 --ranges 0-11,200-202
  FLUSH=0 … # выключить вытеснение собранных текстов из кэша движка (по умолчанию включено)

Границы режутся по абзацам; если абзацы крупнее допуска, рез становится мельче
(одинарные переносы, затем посимвольно), а фактическое число токенов пишется в
manifest — больше 5 % отклонения от таргета считается ошибкой, а не «ну почти».
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
ORDER = ["war_and_peace", "moby_dick", "dracula", "monte_cristo",
         "don_quixote", "les_miserables"]


def parse_ranges(spec):
    slots = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            slots.extend(range(int(a), int(b) + 1))
        else:
            slots.append(int(part))
    return slots


def strip_gutenberg(t):
    m = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", t)
    if m:
        t = t[m.end():]
    m = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", t)
    if m:
        t = t[:m.start()]
    return t.strip()


def make_counter(url):
    def count(text):
        body = json.dumps({
            "model": "bench-prepare",
            "messages": [{"role": "user", "content": text}],
        }).encode()
        req = urllib.request.Request(
            f"{url}/v1/messages/count_tokens", data=body,
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
    return count


def flush_cache(url, count):
    """Вытеснить собранные чанки из контекст-кэша движка (см. prepare)."""
    words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
             "kilo lima mike november oscar papa quebec romeo sierra tango "
             "uniform victor whiskey xray yankee zulu").split()
    import random
    r = random.Random(99)
    txt = " ".join(r.choice(words) for _ in range(160_000))
    n = count(txt)
    txt = txt[:int(len(txt) * min(1.0, 235_000 / max(n, 1)))]
    body = json.dumps({"model": "bench-prepare",
                       "messages": [{"role": "user",
                                     "content": f"[flush] {txt}"}],
                       "max_tokens": 1}).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            resp.read()
        print("engine context cache flushed (unrelated 235k-token request)")
    except Exception as e:
        print(f"WARN: flush failed ({e}) — первый прогон может дать cache hit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("URL", "http://localhost:8082"))
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR",
                      os.path.join(ROOT, "realdata")))
    ap.add_argument("--depth", type=int, required=True, help="токен-таргет чанка")
    ap.add_argument("--ranges", required=True,
                    help="слоты: '0-2,200-202' (имена файлов <depth>-<slot>.txt)")
    ap.add_argument("--tol", type=float, default=0.005, help="допуск по токенам")
    args = ap.parse_args()

    url = args.url.rstrip("/")
    data = args.data_dir
    man_path = os.path.join(data, "manifest.json")
    manifest = json.load(open(man_path, encoding="utf-8")) if os.path.exists(man_path) else {}
    count = make_counter(url)

    books = {}
    for fn in sorted(os.listdir(os.path.join(data, "src"))):
        if fn.endswith(".txt"):
            books[fn[:-4]] = strip_gutenberg(
                open(os.path.join(data, "src", fn), encoding="utf-8",
                     errors="replace").read())

    # Курсоры: за край ВСЕГО, что уже отдавали под чанки той же книги. Считается
    # по файлам на диске, а не по манифесту: прошлая сборка могла упасть до
    # записи манифеста, и тогда её тексты остались на диске неучтёнными —
    # следующий прогон дал бы по ним cache hit (ровно то, мы и лечим).
    cursor = {b: 0 for b in books}
    used = 0
    for fn in sorted(os.listdir(os.path.join(data, "chunks"))):
        if not fn.endswith(".txt"):
            continue
        text = open(os.path.join(data, "chunks", fn), encoding="utf-8").read()
        hit = next(((b, books[b].find(text[:2000])) for b in ORDER
                    if b in books and books[b].find(text[:2000]) != -1), None)
        if hit:
            b, pos = hit
            cursor[b] = max(cursor[b], pos + len(text))
            used += 1
    for key, meta in manifest.items():
        b = meta.get("book")
        if b in cursor:
            cursor[b] = max(cursor[b], meta.get("char_start", 0) + meta.get("chars", 0))
    print(f"cursors ({used} чанков на диске учтено):",
          " ".join(f"{b}@{cursor[b]:,}" for b in ORDER if b in cursor))

    def cut_at(t, pos, brk):
        """Граница после pos. brk="" — резать по символу (последняя инстанция)."""
        ln = t.find(brk, pos)
        return ln + len(brk) if ln != -1 else None

    def bisect(t, start, target, brk, hi):
        lo, best = start + 1, None
        for _ in range(16):
            mid = (lo + hi) // 2
            end = cut_at(t, mid, brk) or hi
            n = count(t[start:end])
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (end, n)
            if abs(n - target) <= target * args.tol:
                break
            if n < target:
                lo = mid + 1
            else:
                hi = mid - 1
            if hi <= lo:
                break
        return best

    def build(book, target):
        """Чанк из cursor[book] размером ~target токенов.

        ("ok", (text, start, n))      — попали в допуск
        ("short", None)               — до конца книги не набрать (книгу можно
                                        считать кончившейся)
        ("coarse", (end, n))          — набрать можно, но границы реза крупнее
                                        допуска: абзацы в некоторых местах идут
                                        шагами ~1000 символов (~260 токенов),
                                        цель лежит между соседними резами. Это
                                        НЕ конец книги: пробуем другие книги, а
                                        если ничего лучше нет — берём ближайшее
                                        и живём с этим процентом.
        """
        t = books[book]
        start = cut_at(t, cursor[book], "\n\n")
        if start is None or start >= len(t) - 1000:
            return ("short", None)
        probe_end = min(start + 200_000, len(t))
        if probe_end - start < 1000:
            return ("short", None)
        cpt = (probe_end - start) / max(count(t[start:probe_end]), 1)
        hi = min(len(t) - 1, int(start + target * 1.8 * cpt))
        if count(t[start:hi]) < target * 1.02:
            return ("short", None)
        end = n = None
        for brk in ("\n\n", "\n", ""):
            end, n = bisect(t, start, target, brk, hi)
            if abs(n - target) <= target * 0.02:
                return ("ok", (t[start:end], start, n))
        return ("coarse", (end, n))

    def save(slot, book, text, cstart, ntok):
        manifest[f"chunks/{args.depth}-{slot}.txt"] = {
            "target_tokens": args.depth, "tokens": ntok, "chars": len(text),
            "book": book, "char_start": cstart,
            "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
            "built_by": "make-chunks.py",
        }
        # на каждый чанк: падение посреди сборки не должно оставлять тексты
        # на диске без учёта (см. курсоры выше)
        json.dump(manifest, open(man_path, "w"), indent=1)

    slots = parse_ranges(args.ranges)
    todo = [s for s in slots
            if not os.path.exists(os.path.join(data, "chunks", f"{args.depth}-{s}.txt"))]
    print(f"слотов {len(slots)}, уже есть {len(slots)-len(todo)}, строю {len(todo)}")
    worst = 0.0
    for i, slot in enumerate(todo):
        # книги по убыванию остатка: round-robin умирал на короткой dracula,
        # когда от войны и мира ещё полкниги лежало нетронутыми
        picked = None                       # (book, (text|end, start, n))
        for book in sorted(ORDER, key=lambda b: -(len(books[b]) - cursor[b])):
            if len(books[book]) - cursor[book] < args.depth * 4:
                continue                    # заведомо мало, не дёргаем движок
            kind, payload = build(book, args.depth)
            if kind == "ok":
                picked = (book, payload)
                break
            if kind == "short":
                cursor[book] = len(books[book])
                print(f"  {book}: остатка не хватает, беру следующую", flush=True)
            elif picked is None or abs(payload[1] - args.depth) < abs(picked[1][1] - args.depth):
                picked = (book, payload)    # coarse: помним, но ищем лучше
        if picked is None:
            sys.exit(f"не хватает уникального текста для слота {slot} "
                     f"(глубина {args.depth}) — книги кончились")
        book, payload = picked
        if len(payload) == 3 and isinstance(payload[0], str):
            text, cstart, ntok = payload
        else:                               # coarse: берём тот же рез, что считался
            end, ntok = payload
            cstart = cut_at(books[book], cursor[book], "\n\n")
            text = books[book][cstart:end]
        off = abs(ntok - args.depth) / args.depth
        worst = max(worst, off)
        if off > 0.05:
            sys.exit(f"чанк {slot} вышел {ntok} токенов при таргете {args.depth} "
                     f"(>5 %) — сравнивать глубину нельзя")
        cursor[book] = cstart + len(text)
        open(os.path.join(data, "chunks", f"{args.depth}-{slot}.txt"), "w",
             encoding="utf-8").write(text)
        save(slot, book, text, cstart, ntok)
        flag = f"  off {off*100:+.1f}%" if off > 0.005 else ""
        print(f"{args.depth}-{slot:<5} {ntok:>7,} tok  book={book:<14} "
              f"chars@{cstart:,}  ({i+1}/{len(todo)}){flag}", flush=True)
    print(f"manifest: {man_path}  худшее отклонение по глубине: {worst*100:.1f} %")

    if os.environ.get("FLUSH", "1") == "1":
        flush_cache(url, count)


if __name__ == "__main__":
    main()
