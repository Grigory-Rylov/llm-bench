#!/usr/bin/env python3
"""Параллельный декод-бенчмарк для Maya (Strata GLM) на :8082.

N одновременных стриминговых запросов, метрики на поток и агрегат:
  ttft, decode tok/s (client и engine), MTP accept, суммарный tok/s
и сравнение с одиночным прогоном (--solo) — чтобы видеть, даёт ли
--parallel реальный прирост агрегата и сколько теряет каждый поток.

Промпты: реальный текст из realdata/chunks (как у bench-real-prefill),
иначе — солевой филлер. У каждого потока свой чанк/соль, чтобы кэш
не обслуживал второй запрос текстом первого.

Примеры:
  ./bench-parallel-decode.py --concurrency 2 --runs 3
  ./bench-parallel-decode.py --solo --runs 2          # базовая линия 1 поток
  ./bench-parallel-decode.py --concurrency 2 --depth 20000
  ./bench-parallel-decode.py --concurrency 2 --verbose
"""
import argparse
import concurrent.futures as cf
import http.client
import json
import os
import random
import string
import statistics as s
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))


def salt():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + \
        "".join(random.choices(string.ascii_lowercase, k=6))


def get_model(url):
    with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]


def load_chunk(data_dir, depth, idx, n_chunks):
    """Чанк для потока idx: отдельный файл на поток, чтобы тексты не пересекались."""
    if not data_dir:
        return None
    letters = "abcdefghijklmnopqrstuvwxyz"
    name = f"{depth}-{letters[idx % len(letters)]}.txt"
    path = os.path.join(data_dir, "chunks", name)
    return open(path, encoding="utf-8").read() if os.path.exists(path) else None


def filler(depth):
    line = ("filler text for the parallel decode benchmark, plain on purpose. ")
    return line * (depth // len(line) + 1)


def build_prompt(text, depth, decode_tokens):
    if text is None:
        text = filler(depth)
    return (f"[bench {salt()}] Read the following text, then CONTINUE it in the "
            f"same style and voice, picking up exactly where it stops. Write at "
            f"least {int(decode_tokens * 1.5)} words. Do not stop early, do not "
            f"summarize, do not comment.\n\n" + text)


def stream_one(url, model, prompt, max_tokens, timeout):
    """Один стриминговый запрос. Возвращает метрики потока."""
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    t0 = time.perf_counter()
    try:
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json",
                              "Content-Length": str(len(body))})
        r = conn.getresponse()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.read(300).decode('utf-8', 'replace')}")
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
                if (d.get("content") or d.get("reasoning_content")) and ttft is None:
                    ttft = time.perf_counter() - t0
        wall = time.perf_counter() - t0
    finally:
        conn.close()
    if ttft is None:
        raise RuntimeError("no token received")
    if usage is None:
        raise RuntimeError("no usage in stream")
    comp = usage.get("completion_tokens", 0)
    dw = wall - ttft
    m = {
        "ttft": ttft,
        "wall": wall,
        "comp": comp,
        "client_dps": comp / dw if dw > 0.05 else 0.0,
        "engine_dps": 0.0,
        "acc_pct": None,
        "short": comp < max_tokens * 0.9,
    }
    if timings:
        pn, pms = timings.get("predicted_n"), timings.get("predicted_ms")
        if pn and pms:
            m["engine_dps"] = pn / (pms / 1000.0)
        dn, da = timings.get("draft_n"), timings.get("draft_n_accepted")
        if dn:
            m["acc_pct"] = (da or 0) / dn * 100
    return m


def run_batch(url, model, prompts, max_tokens, timeout, label, verbose):
    """Параллельный прогон: все промпты одновременно. Возвращает метрики."""
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        futs = [ex.submit(stream_one, url, model, p, max_tokens, timeout)
                for p in prompts]
        results = []
        for i, f in enumerate(futs):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  {label}-{i+1}  FAIL: {e}")
                results.append(None)
    agg_wall = time.perf_counter() - t0
    ok = [m for m in results if m]
    if not ok:
        return None, agg_wall
    # агрегат: сумма engine-токенов / время самого медленного потока
    total_tok = sum(m["comp"] for m in ok)
    agg = total_tok / agg_wall if agg_wall > 0 else 0.0
    for i, m in enumerate(results):
        if not m:
            continue
        dps = m["engine_dps"] or m["client_dps"]
        tag = "" if m["engine_dps"] else " (client)"
        acc = f"  MTP acc={m['acc_pct']:.0f}%" if m["acc_pct"] is not None else ""
        short = "  ⚠ SHORT-STOP" if m["short"] else ""
        if verbose:
            print(f"  {label}-{i+1}  decode={dps:6.1f}{tag} tok/s  "
                  f"ttft={m['ttft']*1000:7.0f}ms  comp={m['comp']}{acc}{short}")
        else:
            print(f"  {label}-{i+1}  decode={dps:6.1f}{tag} tok/s  "
                  f"ttft={m['ttft']*1000:7.0f}ms{acc}{short}")
    dps_agg = [m["engine_dps"] for m in ok if m["engine_dps"]] or \
              [m["client_dps"] for m in ok]
    accs = [m["acc_pct"] for m in ok if m["acc_pct"] is not None]
    print(f"  [{label}] aggregate={agg:6.1f} tok/s  "
          f"(сумма потоков={sum(dps_agg):6.1f})  "
          f"ttft max={max(m['ttft'] for m in ok)*1000:.0f}ms")
    if accs:
        print(f"  [{label}] MTP accept  mean={s.mean(accs):5.1f}%")
    return {"agg": agg, "sum": sum(dps_agg), "results": ok}, agg_wall


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("URL", "http://localhost:8082"))
    ap.add_argument("--model", default=None, help="по умолчанию из /v1/models")
    ap.add_argument("--concurrency", type=int, default=2, help="потоков одновременно")
    ap.add_argument("--runs", type=int, default=3, help="прогонов на каждую конфигурацию")
    ap.add_argument("--depth", type=int, default=4000,
                    help="размер промпта в словах-заполнителях (или глубина чанка)")
    ap.add_argument("--max-tokens", type=int, default=128, help="цель декода")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "realdata"),
                    help="папка с realdata/chunks (пусто = филлер)")
    ap.add_argument("--chunk-depth", type=int, default=None,
                    help="глубина чанков реального текста (по умолчанию = --depth)")
    ap.add_argument("--solo", action="store_true",
                    help="сначала базовая линия: 1 поток, те же прогоны")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--verbose", action="store_true", help="добавить comp в строки")
    args = ap.parse_args()

    model = args.model or get_model(args.url)
    data_dir = args.data_dir if os.path.isdir(
        os.path.join(args.data_dir, "chunks")) else None
    chunk_depth = args.chunk_depth or args.depth

    print(f"model   {model}")
    print(f"url     {args.url}")
    print(f"потоков {args.concurrency}  прогонов {args.runs}  "
          f"max_tokens={args.max_tokens}  "
          f"промпт={'реальные чанки ' + str(chunk_depth) if data_dir else 'филлер ' + str(args.depth)}")

    # прогрев: один маленький запрос, чтобы движок и сети проснулись
    try:
        stream_one(args.url, model,
                   f"[warm {salt()}] Say the single word READY.", 4, args.timeout)
    except Exception as e:
        print(f"WARN: warmup failed: {e}")

    solo_base = None
    if args.solo:
        print(f"\n=== solo (1 поток, {args.runs} прогонов) ===")
        solo_all = []
        for run in range(args.runs):
            text = load_chunk(data_dir, chunk_depth, run, 26) if data_dir else None
            prompt = build_prompt(text, args.depth, args.max_tokens)
            r, _ = run_batch(args.url, model, [prompt], args.max_tokens,
                             args.timeout, f"solo-{run+1}", args.verbose)
            if r:
                solo_all.append(r["agg"])
        if solo_all:
            solo_base = s.mean(solo_all)
            print(f"\n  solo baseline: aggregate mean={solo_base:.1f} tok/s "
                  f"(n={len(solo_all)})")

    print(f"\n=== parallel x{args.concurrency} ({args.runs} прогонов) ===")
    par_all, par_sums = [], []
    for run in range(args.runs):
        prompts = []
        for p in range(args.concurrency):
            # у каждого потока свой чанк: файлы -a -b -c... не пересекаются
            text = load_chunk(data_dir, chunk_depth,
                              run * args.concurrency + p, 26) if data_dir else None
            prompts.append(build_prompt(text, args.depth, args.max_tokens))
        r, _ = run_batch(args.url, model, prompts, args.max_tokens,
                         args.timeout, f"par-{run+1}", args.verbose)
        if r:
            par_all.append(r["agg"])
            par_sums.append(r["sum"])
    if not par_all:
        sys.exit("нет успешных прогонов")

    print(f"\n=== summary [parallel x{args.concurrency}] (n={len(par_all)}) ===")
    print(f"  aggregate     mean={s.mean(par_all):8.1f} tok/s"
          + (f"  min={min(par_all):.1f}  max={max(par_all):.1f}" if len(par_all) > 1 else ""))
    if solo_base:
        gain = s.mean(par_all) / solo_base
        print(f"  vs solo       {solo_base:.1f} -> {s.mean(par_all):.1f} tok/s  "
              f"x{gain:.2f} ({(gain-1)*100:+.0f}%)")
        print(f"  на поток      {s.mean(par_all)/args.concurrency:.1f} tok/s "
              f"(одиночный давал {solo_base:.1f})")
    if par_sums:
        print(f"  сумма потоков mean={s.mean(par_sums):8.1f} tok/s "
              f"(aggregate ниже — потоки ждут друг друга на сплите)")


if __name__ == "__main__":
    main()
