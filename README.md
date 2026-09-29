# llm-depth-bench

Measure **prefill and decode throughput against context depth** on an
OpenAI-compatible LLM server that is **already running** — one you did not
launch, may not be able to restart, and may not even be on this machine.

**日本語 → [README.ja.md](README.ja.md) ・ 简体中文 → [README.zh-CN.md](README.zh-CN.md)**

It reimplements the measurement protocol and the output schema of
[llama-split-bench](https://github.com/kuraneko1/llama-split-bench) (kuraneko1,
MIT), so the JSON it writes can be plotted by that project's `plot_bench.py`
without modification and the numbers line up with reports produced by it.

## Why this exists

llama-split-bench starts its own llama.cpp `llama-server` and reads the
`timings` field that `/completion` returns. Neither is available when the thing
you want to measure is a long-lived vLLM deployment: the server is already up
(often serving other people), and the OpenAI-compatible API has no `timings`.

This tool derives the same quantities on the client side from a single streaming
completion:

| llama.cpp `timings` | how it is obtained here |
|---|---|
| `prompt_per_second` | (prompt tokens newly evaluated this turn) / TTFT |
| `predicted_per_second` | `completion_tokens` / (total time − TTFT) |
| `cache_n` | delta of `prefix_cache_hits_total` in `/metrics` |
| `draft_n` / `draft_n_accepted` | delta of `spec_decode_*` in `/metrics` |

TTFT is the wall-clock time until the first token of the stream arrives, so the
prefill figure includes network and scheduling overhead. On a loopback
connection to an idle server that overhead is small, but it is not zero, and it
is the reason the numbers are attributed to "the server as deployed" rather than
to the GPU alone.

## What it measures

1. **Depth ladder, reusing context.** One prompt that grows stage by stage
   (0 → 32k → … → target). Each stage reports the *incremental* prefill rate and
   then generates N tokens (`ignore_eos`, default 1000) for the decode rate.
   This is the shape of real agent use: a long prompt, then generation deep in
   the context.
2. **Prefill at depth 0.** Fresh prompts of 512 / 2048 / 8192 tokens with the
   prefix cache isolated, giving the correct prefill rate on an empty context.
   The ladder's first stage is an 11-token prompt, so *its* prefill number is a
   measurement artifact and must not be plotted.
3. **Speculative-decoding acceptance.** With a draft model or MTP head, decode
   speed depends on how many drafted tokens are accepted. Synthetic ladder text
   is accepted at ~0.9, which flatters decode badly.
4. **Real-prompt correction.** Three workload-proxy prompts at temperature 0.7
   are run separately to produce a correction factor. See the warning below.
5. **Provenance.** `run-info.json` records what the server says about itself,
   where the launch arguments came from, GPU telemetry sampled throughout the
   run, and the speculative-decoding shape recovered from the metrics.

## Requirements

- Python 3.8+ — **standard library only**. No pip install.
- A reachable OpenAI-compatible server with streaming `/v1/completions`.
- Optional, for figures: `matplotlib` and `plot_bench.py` from llama-split-bench.
- Optional, for GPU telemetry: an NVIDIA driver (NVML is used directly; see below).

## Quick start

```bash
git clone https://github.com/<you>/llm-depth-bench && cd llm-depth-bench
mkdir -p runs

./llm-depth-bench.py \
  --tag my-first-run \
  --url http://127.0.0.1:8000 \
  --model my-served-model-name \
  --mode tp2 \
  --devices CUDA0,CUDA1 \
  --guard-devices 0,1 \
  --ctx 262144 \
  --stages 0,32000,64000,128000,196000,258000 \
  --n-predict 1000 \
  --cache-salt auto \
  --machine "some neutral description of the box"
```

If the server requires a key, pass `--api-key` or set
`LLM_DEPTH_BENCH_API_KEY` (or `OPENAI_API_KEY`) in the environment.

A deep ladder takes tens of minutes. Run it detached and poll the log — the
script prints `BENCH-DONE <tag> <time>` on success and `BENCH-ABORT: …` on
failure, so no interactive terminal is needed:

```bash
setsid nohup ./llm-depth-bench.py --tag my-first-run … > runs/my-first-run.log 2>&1 &
grep -E 'BENCH-(DONE|ABORT)' runs/my-first-run.log
```

Depth-resolved decode with realistic generation (separates the depth effect from
the acceptance effect):

```bash
./depth_real.py --url http://127.0.0.1:8000 --model my-served-model-name \
  --stages 0,32000,128000,258000 --out runs/my-first-run/results-real-depth.json
```

## Figures

Figures come from llama-split-bench's own `plot_bench.py`, so a report made with
this tool looks like one made with that tool:

```bash
python plot_bench.py --dir runs/my-first-run --series tp2 --lang en --estimate off
```

`plot_bench.py` hardcodes "llama.cpp" in the subtitle; the one-line
[`plot_bench-engine_label.patch`](plot_bench-engine_label.patch) in this
repository makes it prefer the `engine_label` field that `run-info.json`
carries. Passing `--series a,b` plots two runs against each other, which is how
you compare two server configurations.

## What is engine-agnostic and what is not

The timing measurement needs only streaming `/v1/completions`, so it works
against any OpenAI-compatible server. Everything that sharpens the measurement
currently speaks vLLM:

| Feature | Preferred source | Fallback when absent |
|---|---|---|
| Timing (prefill, decode) | `POST /v1/completions` (stream) | — required, works everywhere |
| Authentication | `--api-key`, `$LLM_DEPTH_BENCH_API_KEY`, `$OPENAI_API_KEY` | sent as `Authorization: Bearer` when set |
| Prompt sizing | `POST /tokenize` (`count`, or `len(tokens)` for llama.cpp) | **automatic**: `usage.prompt_tokens` from a `max_tokens=1` completion. Works anywhere, but each probe is a real prefill |
| Cache attribution | a `prefix_cache_hits_total` metric | column absent; prefill is then reported against the whole prompt, which over-reports it wherever the cache was hit |
| Acceptance rate | `spec_*_draft_tokens_total` / `..._accepted_tokens_total` | column absent |
| Contention check | a `num_requests_running` / `num_running_reqs` metric | no warning when another request is in flight |
| Per-run cache isolation | `cache_salt` request field | **automatic**: the salt is prefixed to the prompt instead |
| GPU telemetry | NVML | `nvidia-smi`, else omitted |

Metric names are resolved by pattern against the bare name, so an engine that
publishes the same concept under its own prefix is picked up without a code
change. Nothing is guessed: whatever a logical metric bound to is written to
`run-info.json` as `metric_names`, and a concept that matched nothing is simply
absent. **Check that binding before trusting a derived column** — it is how a
mis-binding to a per-position breakdown was caught during development.

**Verified on:** vLLM `0.1.dev20073` on Linux with NVIDIA GPUs, with the full
feature set; and against an authenticated OpenAI-compatible proxy with no
`/tokenize`, no `/metrics` and no `/version`, where the timing columns are
produced and the rest degrade to absent. llama.cpp's `llama-server` and SGLang
have **not** been tried; the fallbacks are written for them but unverified.

## Gotchas worth knowing before you trust a number

- **Always pass `--cache-salt auto` when you intend to repeat a run.** Without
  it, a second run of the same ladder hits the prefix cache: the 32k stage
  evaluates a couple of thousand new tokens instead of 32,000 and reports a
  completely different quantity that looks like a plausible prefill rate. This
  is the single easiest way to fool yourself with this tool.
- **Never plot the ladder's depth-0 prefill.** It is an 11-token prompt.
- **The real-prompt correction factor is unreliable for speculative models.**
  llama-split-bench computes it as (real-prompt decode) / (ladder depth-0
  decode), but depth 0 is too short for the draft to warm up, so the factor can
  exceed 1. Generate the figure with `--estimate off` when that happens.
- **Decode depends on what you generate, not just how deep you are.** On one
  MTP model the same depth gave 178 tok/s on synthetic ladder text (acceptance
  0.93) and 70 tok/s on realistic generation (acceptance 0.21). Always report
  the acceptance rate next to a decode number.
- **`results-real-depth.json` is one sample per depth at temperature 0.7.**
  Treat it as ±30%.
- **Measuring a shared server is possible but not free.** Each stage checks
  `num_requests_running` first and warns, it does not block.
- **Server uptime is a variable.** Two runs against the same hardware minutes
  and days after a restart are not automatically comparable.
- **A proxy that silently drops unknown fields defeats cache isolation.** The
  `cache_salt` probe only sees whether the request was accepted, so a gateway
  configured to discard parameters it does not recognise reports
  `cache_salt_mode: field` while the salt never reaches the engine. If repeats
  of a ladder show suspiciously small `prompt_n`, the salt is not arriving.

## Output

Everything lands in `runs/<tag>/`. Files at the top level are safe to publish;
generated text goes one level down in `responses/` so it can be excluded by
directory rather than by filename.

| File | Contents |
|---|---|
| `results-<mode>.json` | ladder: one record per stage |
| `results-<mode>-pp0.json` | prefill on an empty context |
| `results-real.json` | real-prompt correction |
| `results-real-depth.json` | depth-resolved realistic decode (`depth_real.py`) |
| `run-info.json` | provenance (see below) |
| `argv-<mode>.txt` | server launch arguments, if they could be found |
| `gpu-telemetry-<mode>.csv` | every GPU sample taken during the run |
| `responses/*.txt` | generated text — **not** intended for publication |

`run-info.json` carries, besides the llama-split-bench fields:

- `server_config` — the engine's **resolved** configuration, read over the API:
  version, models and `max_model_len`, every Prometheus `*_info` metric (for
  vLLM that is the whole KV-cache configuration), and the speculative shape.
  This is better evidence than a command line, which only records what was
  *asked for*.
- `server_config.speculative` — the draft length recovered two independent ways
  (draft tokens ÷ drafts, and the number of `position` labels), plus how often
  each draft position is accepted.
- `argv_source` — how the launch arguments were obtained, or why they are
  missing: `ps` on Linux/macOS, `Get-CimInstance Win32_Process` on Windows,
  `/proc` as a last resort, or `--argv-file` for a remote server.
- `metric_names` — which concrete metric each logical metric bound to.
- `token_count_backend` — `endpoint` or `usage` (see the table above).
- `cache_salt_mode` — `field` or `prefix`.
- `gpu_probe_backend`, `gpu_samples`, `tokenize_calls`.

Per-stage records include `gpu_window`: min / mean / max of power, SM and memory
clock, utilisation and temperature over that stage's own time window. Note the
window starts just before the request, so the minimum may be an idle sample;
read the maximum to answer "was it power limited?".

## GPU telemetry

Sampling goes through NVML directly (`libnvidia-ml.so.1`, or `nvml.dll` on
Windows) using `ctypes`, falling back to parsing `nvidia-smi` where the library
cannot be loaded. Both report identical values — `nvidia-smi` is itself an NVML
client — but NVML costs about 0.75 ms per sample against about 61 ms for forking
`nvidia-smi`, so the sampler can run every second without becoming part of what
it measures. Set `--gpu-sample-interval 0` to disable.

## Platform support

Linux is the only platform this has been run on end to end. Nothing in the code
is Linux-only: NVML loads on Windows, process discovery uses `ps` on
Linux/macOS and a CIM query on Windows (`wmic` is deliberately not used — it has
been removed from current Windows), and all file writes are explicitly UTF-8.
Reports of it working or failing elsewhere are welcome.

## Credit

The measurement protocol, the output schema, and the figures are
[llama-split-bench](https://github.com/kuraneko1/llama-split-bench) by
kuraneko1, MIT licensed. That project is the right tool when you can launch the
server yourself: it answers "layer split or tensor split?" end to end, which
this one deliberately does not attempt.

## License

MIT — see [LICENSE](LICENSE).
