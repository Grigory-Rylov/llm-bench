# llm-bench

Бенчмарки prefill/decode для локальных LLM-серверов (Strata, llama.cpp-совместимые
OpenAI/Anthropic-API) на реальном тексте.

Два инструмента:

| Скрипт | Что мерит |
|---|---|
| `bench-real-prefill.sh` | TTFT и скорость prefill на глубинах 40k/150k/200k токенов + опциональный decode-пробник |
| `bench-parallel-decode.py` | Параллельное декодирование: N одновременных стримов, прирост агрегата vs соло |

Оба кормят движок **настоящей прозой** (public-domain книги, см. `realdata/src/`),
а не повторённой фразой: каждый прогон получает свой непересекающийся диапазон
текста, поэтому ни один замер не может быть обслужен prefix/context-кэшем после
другого прогона.

## Быстрый старт

По умолчанию все скрипты стучатся на движок `http://localhost:8082`.
Чтобы гонять на произвольном хосте/порту, передай `URL`:

```bash
# другой порт на этой машине
URL=http://localhost:8085 ./bench-real-prefill-prepare.sh
URL=http://localhost:8085 ./bench-real-prefill.sh
URL=http://localhost:8085 ./bench-parallel-decode.py --concurrency 2

# движок на другой машине
URL=http://192.168.1.50:8080 ./bench-real-prefill.sh
```

Стандартный запуск (движок на :8082):

```bash
# 1. Сгенерировать чанки (нужен запущенный движок — размеры считаются его токенизатором)
./bench-real-prefill-prepare.sh

# 2. Prefill-бенч (по умолчанию глубины 40k/150k/200k, 3 прогона на глубину)
./bench-real-prefill.sh

# 3. Параллельный декод: 2 потока против соло-базы
./bench-parallel-decode.py --solo --runs 2
./bench-parallel-decode.py --concurrency 2 --runs 3
```

## Требования

- Linux, bash, python3 (stdlib, без зависимостей)
- запущенный движок с OpenAI-совместимым `/v1/chat/completions` (stream)
  и, для prepare, Anthropic-совместимым `/v1/messages/count_tokens` (Strata)
- по умолчанию движок на `http://localhost:8082`

## bench-real-prefill-prepare.sh

Готовит `realdata/`: `src/` (скачивает книги с Gutenberg, если их нет),
`chunks/<DEPTH>-{a,b,c}.txt` (prefill) и `chunks/<DEPTH>-d{1,2,3}.txt` (decode),
`manifest.json`.

Чанки режутся **токенами движка** (через `POST /v1/messages/count_tokens`),
курсор по книгам движется — все чанки всех глубин взаимно непересекающиеся,
поэтому ни один прогон не обслуживается контекст-кэшем после другого.
`manifest.json` фиксирует токены/sha256/книгу/смещение каждого чанка.

Env-переменные:

| Переменная | Default | Значение |
|---|---|---|
| `URL` | `http://localhost:8082` | базовый URL движка (для count_tokens) |
| `DATA_DIR` | `<скрипт>/realdata` | куда писать |
| `DEPTHS` | `40000,150000,200000` | CSV целей по токенам |
| `FORCE` | `0` | `1` — пересобрать чанки при живом manifest |

Книги (ID Gutenberg): War and Peace 2600, Moby Dick 2701, Dracula 345,
Don Quixote 996, Monte Cristo 1184, Les Misérables 135 — см. массив `BOOKS`
в скрипте. Хедрум ~2.3M уникальных токенов — хватает на 3×(40k+150k+200k)
prefill + 3×decode на каждую глубину.

## bench-real-prefill.sh

Для каждой глубины: warmup того же размера (первый большой запрос после простоя
— выброс), затем `RUNS` холодных prefill-запросов (инструкция «reply DONE»,
max_tokens=16) и decode-пробник (продолжение текста, ~128 токенов).

Метрики: TTFT (клиент), engine tok/s (`timings.prompt_n/prompt_ms` — чистый
prefill-compute), client tok/s (`prompt_tokens/TTFT`, с оверхедом), decode
tok/s клиентский и движковый, MTP accept. Каждый прогон проверяется на
контаминацию кэшем (`cached_tokens`, `cache_n`) и помечается при ненулевом
попадании.

Env-переменные:

| Переменная | Default | Значение |
|---|---|---|
| `URL` | `http://localhost:8082` | движок |
| `MODEL` | из `/v1/models` | id модели |
| `DATA_DIR` | `<скрипт>/realdata` | чанки |
| `DEPTHS` | `40000,150000,200000` | глубины контекста |
| `RUNS` | `3` | прогонов на глубину (каждый — свой чанк) |
| `MAX_TOKENS` | `16` | лимит длины ответа в prefill-запросе (меряются TTFT/prefill, ответ — одно слово «DONE») |
| `DECODE_PROBE` | `1` | `0` — выключить decode-пробник |
| `DECODE_TOKENS` | `128` | цель длины продолжения |
| `DECODE_RUNS` | `3` | прогонов пробника (чанки d1..d3) |
| `CLIENT` | `0` | `1` — печатать и клиентские скорости |
| `QUIET` | `1` | `0` — подробно (чанки, кэш-счётчики, std/CV) |

Пример: подробный прогон только 40k с клиентскими скоростями:

```bash
DEPTHS=40000 CLIENT=1 QUIET=0 ./bench-real-prefill.sh
```

## bench-parallel-decode.py

N одновременных стриминговых запросов; метрики на поток и агрегат: TTFT,
decode tok/s (клиент и движок), MTP accept, суммарный tok/s. Промпты — реальные
чанки (`realdata/chunks`), у потока свой чанк и соль.

```bash
./bench-parallel-decode.py --solo --runs 2              # база: 1 поток
./bench-parallel-decode.py --concurrency 2 --runs 3     # 2 потока
./bench-parallel-decode.py --concurrency 4 --depth 20000 --max-tokens 256
./bench-parallel-decode.py --concurrency 2 --verbose    # + completion-метрики
```

Флаги: `--url`, `--model`, `--concurrency`, `--runs`, `--depth` (токенов в
промпт), `--max-tokens`, `--data-dir`, `--chunk-depth` (из какой глубины брать
чанки), `--solo`, `--timeout`, `--verbose`.

Хост/порт движка — `--url` (или env `URL`), по умолчанию `http://localhost:8082`:

```bash
./bench-parallel-decode.py --url http://192.168.1.50:8085 --concurrency 2
URL=http://localhost:8085 ./bench-parallel-decode.py --concurrency 2
```

Смысл: `--parallel` у движка даёт прирост агрегата, но каждый поток платит
долей; сравнение соло vs N показывает реальную выгоду многопоточки.

## Лицензия источников

Книги в `realdata/src/` — public domain (Project Gutenberg).
