#!/usr/bin/env python3
"""llm-depth-bench: measure prefill and decode against context depth, over an
already-running OpenAI-compatible server.

Reimplements the measurement protocol and output schema of llama-split-bench
(kuraneko1, MIT) for servers this tool does not launch. llama-split-bench starts
its own llama.cpp `llama-server` and reads the `timings` field of /completion;
neither is available for a long-lived vLLM deployment, so the same quantities are
derived client-side from a streaming completion:

  prompt_per_second    = (prompt tokens newly evaluated this turn) / TTFT
  predicted_per_second = completion_tokens / (total time - TTFT)
  cache_n              = delta of prefix_cache_hits_total in /metrics
  draft_n / accepted   = delta of spec_decode_* in /metrics

What it produces (the schema plot_bench.py from llama-split-bench reads as-is):

  1. depth ladder, reusing context         -> results-<mode>.json
  2. prefill at depth 0, fresh prompts     -> results-<mode>-pp0.json
  3. speculative-decoding acceptance       -> included in both, from /metrics
  4. real-prompt correction factor         -> results-real.json
  5. provenance                            -> run-info.json / argv-<mode>.txt
                                              gpu-telemetry-<mode>.csv

Backend support: the timing measurement needs only a streaming /v1/completions,
so it is engine-agnostic. Token-exact prompt sizing (/tokenize), cache
attribution, acceptance rates and per-run cache isolation (cache_salt) currently
read vLLM-specific endpoints and metric names; against another OpenAI-compatible
server those columns are absent rather than wrong. See the README.
"""
import argparse
import hashlib
import json
import os
import random
import re
import ctypes
import ctypes.util
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import datetime

TAIL = ("\n\nContinue the sequence with the next 100 numbered lines, in exactly "
        "the same style. Do not stop and do not summarize.")
STAGE0_PROMPT = "The capital of France is and the capital of Germany is"

REAL_PROMPTS = [
    ("design", "次の設計判断を整理して: 単一のV100 32GBで262kコンテキストの推論を回すとき、"
               "KVキャッシュの型・重みの量子化・投機デコードの3つは互いにどう影響し合うか。"
               "トレードオフを表にまとめ、優先順位と根拠を示して。"),
    ("review", "長文の技術文書をレビューする立場で、次の観点を順に検討して: 主張の根拠が測定に基づいているか、"
               "見落とされている交絡因子はないか、結論を出す前に必要な追加検証は何か。"
               "それぞれ具体的な反例を挙げて説明して。"),
    ("qa",     "以下を自分の言葉で説明して: なぜGPUのGEMVは帯域律速になるのか、"
               "なぜ量子化ブロックのメモリ配置が実効帯域を変えるのか、"
               "そして帯域を測る際に論理バイト数と実DRAM転送量がずれるのはどんな時か。"),
]

# Logical metric -> regex matched against the Prometheus metric name, without
# the engine prefix. vLLM is what these have been verified against; the patterns
# are deliberately prefix-agnostic so an engine publishing the same concept under
# its own prefix is picked up too. Nothing is guessed: a concept that matches no
# metric is reported absent, and the name that did match is recorded in
# run-info.json so a reader can check the binding.
METRIC_PATTERNS = {
    # Anchored at the end so a longer, different metric cannot satisfy them:
    # `..._accepted_tokens_per_pos_total` is a per-position breakdown, not the
    # total, and `external_prefix_cache_hits_total` counts an external KV cache.
    "prefix_cache_hits": r"^(?!external_).*prefix_cache_hits_total$",
    "prefix_cache_queries": r"^(?!external_).*prefix_cache_queries_total$",
    "spec_draft_tokens": r"spec\w*draft_tokens_total$",
    "spec_accepted_tokens": r"spec\w*accepted_tokens_total$",
    "spec_drafts": r"spec\w*num_drafts_total$",
    "num_requests_running": r"(num_requests_running|num_running_reqs)$",
    "num_requests_waiting": r"(num_requests_waiting|num_queue_reqs)$",
    "kv_cache_usage": r"kv_cache_usage\w*$",
}


def fill_text(n_chars):
    unit = "Sequence {i}: the quick brown fox jumps over the lazy dog while we measure context tokens. "
    parts, total, i = [], 0, 1
    while total < n_chars:
        p = unit.format(i=i)
        parts.append(p)
        total += len(p)
        i += 1
    return "".join(parts)[:n_chars]


class Server:
    """Thin client for an OpenAI-compatible server.

    Only streaming /v1/completions is required. Everything else -- token
    counting, Prometheus metrics, per-run cache isolation -- is probed once and
    degrades to a documented fallback when the server does not provide it, so
    the core measurement works against engines other than the one this was
    written for.
    """

    def __init__(self, url, model, api_key=""):
        self.url = url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self._tok_memo = {}
        self.tokenize_calls = 0
        self.token_backend = None       # "endpoint" | "usage", set on first use
        self.cache_salt_mode = None     # "field" | "prefix", set by probe_cache_salt
        self._metric_names = {}

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, path, body, timeout=3600):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers=self._headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def _get(self, path, timeout=60):
        req = urllib.request.Request(self.url + path, headers=self._headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    # ---- token counting -------------------------------------------------

    def _count_via_endpoint(self, text):
        """POST /tokenize. vLLM answers with `count`; llama.cpp only with `tokens`."""
        body = self._post("/tokenize", {"model": self.model, "prompt": text}, timeout=300)
        if isinstance(body.get("count"), int):
            return body["count"]
        if isinstance(body.get("tokens"), list):
            return len(body["tokens"])
        raise RuntimeError(f"unrecognised /tokenize response: {sorted(body)[:6]}")

    def _count_via_usage(self, text):
        """Ask the model to generate one token and read usage.prompt_tokens.

        Works against any OpenAI-compatible server, but it is a real prefill:
        counting a 32k prompt this way costs a 32k prefill. Callers should keep
        the number of probes down when this is the active backend.
        """
        body = self._post("/v1/completions",
                          {"model": self.model, "prompt": text, "max_tokens": 1,
                           "temperature": 0}, timeout=3600)
        return body["usage"]["prompt_tokens"]

    def n_tokens(self, text):
        """Token count, memoised by content.

        The sizing loop asks about the same tail at every stage and re-asks for a
        prompt it has already measured, so without memoising the tokenizer sees
        up to eight round trips per stage for at most three distinct strings.
        """
        key = hashlib.sha1(text.encode()).hexdigest()
        if key in self._tok_memo:
            return self._tok_memo[key]
        if self.token_backend is None:
            try:
                n = self._count_via_endpoint(text)
                self.token_backend = "endpoint"
            except Exception as e:
                print(f"note: /tokenize unusable ({type(e).__name__}); counting tokens "
                      f"through usage.prompt_tokens instead, which costs a real prefill "
                      f"per probe", flush=True)
                self.token_backend = "usage"
                n = self._count_via_usage(text)
        elif self.token_backend == "endpoint":
            n = self._count_via_endpoint(text)
        else:
            n = self._count_via_usage(text)
        self.tokenize_calls += 1
        self._tok_memo[key] = n
        return n

    # ---- metrics --------------------------------------------------------

    def _metrics_text(self):
        req = urllib.request.Request(self.url + "/metrics", headers=self._headers())
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode()

    INFO_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z_:0-9]*_info)\{(.*)\}\s+[-+0-9.eE]+$')
    LABEL_RE = re.compile(r'([a-zA-Z_0-9]+)="((?:[^"\\]|\\.)*)"')
    SAMPLE_RE = re.compile(r'^([a-zA-Z_:][^\s{]*)(\{[^}]*\})?\s+([-+0-9.eE]+)$')

    def info_metrics(self):
        """Every Prometheus *_info metric, as {metric: {label: value}}.

        vLLM publishes the engine's resolved configuration here -- what it
        actually decided, not what a command line asked for -- and unlike reading
        the process table this is readable from another host and on any OS.
        """
        out = {}
        for line in self._metrics_text().splitlines():
            m = self.INFO_RE.match(line)
            if m:
                out[m.group(1)] = {k: v for k, v in self.LABEL_RE.findall(m.group(2))}
        return out

    @property
    def metric_names(self):
        """Which concrete metric name each logical metric resolved to."""
        return dict(self._metric_names)

    def metrics(self):
        """{logical name: value}, summed over label sets. Missing concepts are absent."""
        out = {}
        try:
            text = self._metrics_text()
        except Exception:
            return out
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            m = self.SAMPLE_RE.match(line)
            if not m:
                continue
            name, raw = m.group(1), m.group(3)
            bare = name.split(":", 1)[-1]
            for logical, pattern in METRIC_PATTERNS.items():
                bound = self._metric_names.get(logical)
                if bound is not None and bound != name:
                    continue  # a logical metric owns exactly one metric name
                if bound is None and not re.search(pattern, bare):
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    break
                self._metric_names[logical] = name
                out[logical] = out.get(logical, 0.0) + value  # sum over label sets
                break
        return out

    # ---- completion -----------------------------------------------------

    def probe_cache_salt(self):
        """Decide how to isolate one run's prefix cache from the next.

        vLLM takes a `cache_salt` request field. Where that is rejected, the same
        effect is had by prefixing the prompt with the salt: it makes the prefix
        genuinely different, at the cost of a few tokens that the measured depth
        accounts for anyway.
        """
        try:
            self.complete("salt probe", 1, ignore_eos=False, cache_salt="probe-0", timeout=120)
            self.cache_salt_mode = "field"
        except Exception:
            self.cache_salt_mode = "prefix"
            print("note: server rejected the cache_salt field; isolating runs with a "
                  "prompt prefix instead", flush=True)
        return self.cache_salt_mode

    def complete(self, prompt, max_tokens, temperature=0.0, top_p=None,
                 ignore_eos=True, cache_salt=None, timeout=3600):
        """Streaming completion. Returns ttft / total / usage measured client-side."""
        if cache_salt and self.cache_salt_mode == "prefix":
            prompt = f"[run {cache_salt}]\n" + prompt
            cache_salt = None
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True}}
        if ignore_eos:
            body["ignore_eos"] = True
        if top_p is not None:
            body["top_p"] = top_p
        if cache_salt:
            body["cache_salt"] = cache_salt
        req = urllib.request.Request(self.url + "/v1/completions",
                                     data=json.dumps(body).encode(), headers=self._headers())
        t0 = time.perf_counter()
        t_first = None
        usage = None
        text_parts = []
        finish = None
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if obj.get("usage"):
                    usage = obj["usage"]
                ch = obj.get("choices") or []
                if ch:
                    if ch[0].get("text"):
                        if t_first is None:
                            t_first = time.perf_counter()
                        text_parts.append(ch[0]["text"])
                    if ch[0].get("finish_reason"):
                        finish = ch[0]["finish_reason"]
        total = time.perf_counter() - t0
        if usage is None or t_first is None:
            raise RuntimeError(f"no usable timing data (usage={usage}, first_token={t_first})")
        return {"ttft": t_first - t0, "total": total, "usage": usage,
                "text": "".join(text_parts), "finish_reason": finish}


def sized_prompt(srv, target_tokens, ratio, cache=None):
    """Build a prompt of ~target_tokens tokens (filler + TAIL), refining the
    chars-per-token ratio with the server's own tokenizer."""
    if target_tokens in (cache or {}):
        return cache[target_tokens]
    tail_n = srv.n_tokens(TAIL)
    body_target = max(1, target_tokens - tail_n)
    chars = int(body_target * ratio)
    for _ in range(6):
        body = fill_text(chars)
        n = srv.n_tokens(body)
        if abs(n - body_target) <= max(4, int(body_target * 0.002)):
            break
        chars = max(1, int(chars * body_target / max(1, n)))
    prompt = body + TAIL
    got = srv.n_tokens(prompt)
    new_ratio = len(body) / max(1, n)
    if cache is not None:
        cache[target_tokens] = (prompt, got, new_ratio)
    return prompt, got, new_ratio


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class GpuProbe:
    """Read power / clocks / utilisation / temperature per GPU.

    Prefers NVML through ctypes and falls back to parsing `nvidia-smi`. Both
    return the same numbers -- nvidia-smi is itself an NVML client -- but NVML is
    an in-process call at ~0.75 ms against ~61 ms for forking nvidia-smi, so a
    sampler can run often enough to be useful without becoming part of what it
    measures. The library is `libnvidia-ml.so.1` on Linux and `nvml.dll` on
    Windows; where neither loads, the subprocess path keeps working.
    """

    CLOCK_SM, CLOCK_MEM, TEMPERATURE_GPU = 1, 2, 0
    SMI_QUERY = ("nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,"
                 "power.draw,clocks.sm,clocks.mem --format=csv,noheader,nounits")

    def __init__(self):
        self._nvml = None
        self._handles = {}
        try:
            lib = ctypes.util.find_library("nvidia-ml")
            for name in (lib, "libnvidia-ml.so.1", "nvml.dll"):
                if not name:
                    continue
                try:
                    cand = ctypes.CDLL(name)
                except OSError:
                    continue
                if cand.nvmlInit_v2() == 0:
                    self._nvml = cand
                    break
        except Exception:
            self._nvml = None

    @property
    def backend(self):
        return "nvml" if self._nvml else "nvidia-smi"

    def _handle(self, index):
        if index not in self._handles:
            h = ctypes.c_void_p()
            if self._nvml.nvmlDeviceGetHandleByIndex_v2(int(index), ctypes.byref(h)) != 0:
                raise RuntimeError(f"no NVML handle for GPU {index}")
            self._handles[index] = h
        return self._handles[index]

    def read(self, idx):
        """{index: {mem_used_mib, util_pct, temp_c, power_w, sm_mhz, mem_mhz}}"""
        if not idx:
            return None
        if self._nvml is not None:
            try:
                return self._read_nvml(idx)
            except Exception:
                self._nvml = None  # fall back for the rest of the run
        return self._read_smi(idx)

    def _read_nvml(self, idx):
        out = {}
        for i in idx:
            h = self._handle(i)
            power, sm, mem_clk, temp = (ctypes.c_uint() for _ in range(4))
            util, mem = _NvmlUtilization(), _NvmlMemory()
            self._nvml.nvmlDeviceGetPowerUsage(h, ctypes.byref(power))
            self._nvml.nvmlDeviceGetClockInfo(h, self.CLOCK_SM, ctypes.byref(sm))
            self._nvml.nvmlDeviceGetClockInfo(h, self.CLOCK_MEM, ctypes.byref(mem_clk))
            self._nvml.nvmlDeviceGetTemperature(h, self.TEMPERATURE_GPU, ctypes.byref(temp))
            self._nvml.nvmlDeviceGetUtilizationRates(h, ctypes.byref(util))
            self._nvml.nvmlDeviceGetMemoryInfo(h, ctypes.byref(mem))
            out[str(i)] = {"mem_used_mib": mem.used // (1024 * 1024), "util_pct": util.gpu,
                           "temp_c": temp.value, "power_w": round(power.value / 1000.0, 2),
                           "sm_mhz": sm.value, "mem_mhz": mem_clk.value}
        return out

    def _read_smi(self, idx):
        try:
            out = subprocess.run(f"{self.SMI_QUERY} -i {','.join(str(i) for i in idx)}",
                                 shell=True, capture_output=True, text=True, timeout=30).stdout
        except Exception:
            return None
        rows = {}
        for line in out.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 7:
                continue
            rows[f[0]] = {"mem_used_mib": f[1], "util_pct": f[2], "temp_c": f[3],
                          "power_w": f[4], "sm_mhz": f[5], "mem_mhz": f[6]}
        return rows


_PROBE = GpuProbe()


def gpu_state(idx):
    return _PROBE.read(idx)


class GpuSampler:
    """Sample nvidia-smi on a timer for the whole run.

    A per-stage before/after pair cannot say whether the cards were power- or
    clock-limited while the work was happening: both points land when the GPU is
    already idle again. This keeps a background thread sampling throughout, and
    hands each stage the aggregate over its own window.
    """

    FIELDS = ("power_w", "sm_mhz", "mem_mhz", "util_pct", "temp_c")

    def __init__(self, idx, interval=2.0):
        self.idx = list(idx or [])
        self.interval = interval
        self.rows = []
        self._stop = threading.Event()
        self._thread = None

    @property
    def enabled(self):
        return bool(self.idx) and self.interval > 0

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 5)

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def _sample(self):
        rows = _PROBE.read(self.idx)
        if not rows:
            return
        now = time.time()
        for gpu, v in rows.items():
            try:
                self.rows.append((now, gpu, *[float(v[f]) for f in self.FIELDS]))
            except (KeyError, TypeError, ValueError):
                continue

    def window(self, t0, t1):
        """min / mean / max per GPU over [t0, t1], or None if nothing was sampled."""
        picked = [r for r in self.rows if t0 <= r[0] <= t1]
        if not picked:
            return None
        out = {}
        for gpu in sorted({r[1] for r in picked}):
            mine = [r for r in picked if r[1] == gpu]
            out[gpu] = {name: {"min": round(min(v), 1), "mean": round(sum(v) / len(v), 1),
                               "max": round(max(v), 1)}
                        for i, name in enumerate(self.FIELDS)
                        for v in [[r[2 + i] for r in mine]]}
            out[gpu]["samples"] = len(mine)
        return out

    def write_csv(self, path):
        if not self.rows:
            return None
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("unix_time,gpu," + ",".join(self.FIELDS) + "\n")
            for r in self.rows:
                fh.write(f"{r[0]:.1f},{r[1]}," + ",".join(str(x) for x in r[2:]) + "\n")
        return path


def delta(a, b, key):
    va, vb = a.get(key), b.get(key)
    if va is None or vb is None:
        return None
    return vb - va


def server_snapshot(srv):
    """What the running server says about itself, over the API only.

    Always available -- including against a remote server and on Windows, where
    the /proc scan below cannot work. Partial by nature: the engine does not
    publish tensor-parallel size, speculative config or the model path, so this
    supplements the launch arguments rather than replacing them.
    """
    snap = {}
    try:
        snap["version"] = srv._get("/version", timeout=30).get("version", "")
    except Exception as e:
        snap["version_error"] = f"{type(e).__name__}: {e}"[:120]
    try:
        models = srv._get("/v1/models")
        snap["models"] = [{k: m.get(k) for k in ("id", "root", "max_model_len")}
                          for m in models.get("data", [])]
    except Exception as e:
        snap["models_error"] = f"{type(e).__name__}: {e}"[:120]
    try:
        snap["info_metrics"] = srv.info_metrics()
    except Exception as e:
        snap["info_metrics_error"] = f"{type(e).__name__}: {e}"[:120]
    try:
        snap["speculative"] = spec_decode_shape(srv)
    except Exception as e:
        snap["speculative_error"] = f"{type(e).__name__}: {e}"[:120]
    return snap


def spec_decode_shape(srv):
    """Recover the speculative-decoding depth from the counters.

    `--speculative-config` is not published anywhere, but two independent
    readings give the draft length: draft tokens divided by drafts, and the
    number of `position` labels on the per-position acceptance counter. The same
    counter also says how often each draft position is accepted, which is where
    a draft starts going wrong.
    """
    text = srv._metrics_text()
    drafts = draft_tokens = None
    per_pos = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        value = line.split()[-1] if " " in line else None
        bare = line.split("{", 1)[0].split(" ", 1)[0].split(":", 1)[-1]
        if re.search(r"spec\w*num_drafts_total$", bare):
            drafts = float(value)
        elif re.search(r"spec\w*draft_tokens\w*_total$", bare):
            draft_tokens = float(value)
        elif re.search(r"spec\w*accepted_tokens_per_pos\w*_total$", bare):
            m = re.search(r'position="(\d+)"', line)
            if m:
                per_pos[int(m.group(1))] = float(value)
    if drafts is None and not per_pos:
        return {"active": False}
    out = {"active": True}
    if drafts and draft_tokens:
        out["draft_tokens_per_draft"] = round(draft_tokens / drafts, 3)
    if per_pos:
        out["num_speculative_tokens"] = len(per_pos)
        total = sum(per_pos.values()) or 1
        out["accepted_share_by_position"] = {
            str(k): round(per_pos[k] / total, 4) for k in sorted(per_pos)}
    return out


def read_argv(args):
    """Server launch arguments, and where they came from.

    A file the caller points at wins: it is the only option that works for a
    server on another host, in another container, or on Windows. Scanning /proc
    stays as a convenience for the common case of measuring a server on this
    same Linux box.
    """
    if args.argv_file:
        try:
            with open(args.argv_file, encoding="utf-8") as fh:
                return fh.read().strip(), f"--argv-file {args.argv_file}"
        except OSError as e:
            print(f"WARNING: --argv-file unreadable ({e}); falling back to /proc", flush=True)
    for line, source in _local_command_lines():
        if "vllm" in line and " serve " in line:
            return line.strip(), source
    return "", "not found on this host (remote server? pass --argv-file)"


def _local_command_lines():
    """(command line, how it was found) for every process on this machine.

    `ps` covers Linux and macOS in one path; Windows has no `ps`, so it uses a
    CIM query -- `wmic` is gone from current Windows and is deliberately not
    used. /proc remains as a last resort for a container with no `ps` installed.
    """
    if sys.platform == "win32":
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
               "Get-CimInstance Win32_Process | "
               "ForEach-Object { $_.CommandLine }"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        except Exception:
            return
        for line in out.splitlines():
            if line.strip():
                yield line, "Get-CimInstance Win32_Process on this host"
        return
    try:
        out = subprocess.run(["ps", "-eww", "-o", "args="],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        out = ""
    for line in out.splitlines():
        if line.strip():
            yield line, "ps on this host"
    if out.strip() or not os.path.isdir("/proc"):
        return
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            c = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if c.strip():
            yield c, "/proc scan on this host"


def default_engine_label(version, snapshot):
    """Subtitle for the figure, built from what the server reported.

    Whatever the engine did not publish -- tensor-parallel size, expert
    parallelism -- is absent here on purpose: a label is not the place to assert
    a configuration nobody measured. Pass --engine-label to state it yourself.
    """
    parts = [f"vLLM {version}" if version else "server"]
    spec = (snapshot or {}).get("speculative") or {}
    if spec.get("active"):
        n = spec.get("num_speculative_tokens")
        parts.append(f"speculative n={n}" if n else "speculative decoding")
    cache = cache_config(snapshot)
    if cache.get("enable_prefix_caching") == "True":
        parts.append("prefix caching")
    return parts[0] + (" (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else "")


def cache_config(snapshot):
    return ((snapshot or {}).get("info_metrics") or {}).get("vllm:cache_config_info") or {}


def run_ladder(srv, args, outdir, guard_idx, sampler):
    stages = [int(x) for x in args.stages.split(",")]
    results_path = os.path.join(outdir, f"results-{args.mode}.json")
    records = []
    ratio = args.ratio_init
    pcache = {}
    for k, target in enumerate(stages):
        m0 = srv.metrics()
        running = m0.get("num_requests_running", 0)
        if running > 0:
            print(f"stage {target}: WARNING {running:.0f} other request(s) already running "
                  f"on the shared server - measurement may be contended", flush=True)
        if target == 0:
            prompt, want, _ = STAGE0_PROMPT, srv.n_tokens(STAGE0_PROMPT), ratio
        else:
            prompt, want, ratio = sized_prompt(srv, target, ratio, pcache)
        before = gpu_state(guard_idx)
        t0 = time.time()
        try:
            r = srv.complete(prompt, args.n_predict, temperature=0.0, ignore_eos=True,
                             cache_salt=args.cache_salt or None)
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode(errors="replace")
            print(f"stage {target}: HTTP {e.code} ({body}) - retrying at 96% fill", flush=True)
            try:
                prompt, want, ratio = sized_prompt(srv, int(target * 0.96), ratio, pcache)
                r = srv.complete(prompt, args.n_predict, temperature=0.0, ignore_eos=True,
                                 cache_salt=args.cache_salt or None)
            except Exception as e2:
                records.append({"aborted": f"stage {target}: {e2}"})
                json.dump(records, open(results_path, "w", encoding="utf-8"), indent=2)
                return records, f"stage {target}: {e2}"
        except Exception as e:
            records.append({"aborted": f"stage {target}: {e}"})
            json.dump(records, open(results_path, "w", encoding="utf-8"), indent=2)
            return records, f"stage {target}: {e}"
        wall = time.time() - t0
        after = gpu_state(guard_idx)
        m1 = srv.metrics()

        u = r["usage"]
        total_prompt = u["prompt_tokens"]
        cache_hits = delta(m0, m1, "prefix_cache_hits")
        cache_n = int(cache_hits) if cache_hits is not None else None
        prompt_n = total_prompt - cache_n if cache_n is not None else total_prompt
        if prompt_n <= 0:
            prompt_n = total_prompt
            cache_n = 0
        pred_n = u["completion_tokens"]
        gen_s = r["total"] - r["ttft"]
        draft_n = delta(m0, m1, "spec_draft_tokens")
        acc = delta(m0, m1, "spec_accepted_tokens")
        rec = {
            "stage": k,
            "target_tokens": target,
            "shrunk": False,
            "prompt_chars": len(prompt),
            "tokens_evaluated": total_prompt,
            "cache_n": cache_n,
            "prompt_n": prompt_n,
            "effective_depth": total_prompt,
            "prompt_ms": round(r["ttft"] * 1000.0, 3),
            "prompt_per_second": round(prompt_n / r["ttft"], 3) if r["ttft"] > 0 else None,
            "predicted_n": pred_n,
            "predicted_ms": round(gen_s * 1000.0, 3),
            "predicted_per_second": round(pred_n / gen_s, 3) if gen_s > 0 else None,
            "draft_n": int(draft_n) if draft_n is not None else 0,
            "draft_n_accepted": int(acc) if acc is not None else 0,
            "draft_accept_rate": round(acc / draft_n, 3) if draft_n else None,
            "wall_s": round(wall, 1),
            "finish_reason": r["finish_reason"],
            "target_tokens_requested": want,
            "kv_cache_usage_perc_after": m1.get("kv_cache_usage"),
            "other_requests_running_before": running,
            "gpu_before": before,
            "gpu_after": after,
            "gpu_window": sampler.window(t0, time.time()),
        }
        records.append(rec)
        json.dump(records, open(results_path, "w", encoding="utf-8"), indent=2)
        print(json.dumps({kk: rec[kk] for kk in
                          ("stage", "target_tokens", "effective_depth", "cache_n", "prompt_n",
                           "prompt_per_second", "predicted_n", "predicted_per_second",
                           "draft_accept_rate", "wall_s")}, ensure_ascii=False), flush=True)
    return records, None


def run_pp0(srv, args, outdir):
    out = {"tag": args.mode}
    salt_base = "vsb-%d-%d" % (int(time.time()), random.randrange(10 ** 6))
    ratio = args.ratio_init
    pcache = {}
    srv.complete("warmup", 8, temperature=0.0, ignore_eos=False,
                 cache_salt=salt_base + "-warm")
    for size in [int(x) for x in args.pp0_sizes.split(",")]:
        prompt, want, ratio = sized_prompt(srv, size, ratio, pcache)
        m0 = srv.metrics()
        r = srv.complete(prompt, 64, temperature=0.0, ignore_eos=True,
                         cache_salt=f"{salt_base}-pp{size}")
        m1 = srv.metrics()
        u = r["usage"]
        hits = delta(m0, m1, "prefix_cache_hits")
        gen_s = r["total"] - r["ttft"]
        out[f"pp{size}"] = {
            "prompt_n": u["prompt_tokens"],
            "cache_n": int(hits) if hits is not None else None,
            "prompt_ms": round(r["ttft"] * 1000.0, 3),
            "prompt_per_second": round(u["prompt_tokens"] / r["ttft"], 3),
            "predicted_n": u["completion_tokens"],
            "predicted_per_second": round(u["completion_tokens"] / gen_s, 3) if gen_s > 0 else None,
            "wall_s": round(r["total"], 2),
        }
        print(f"{args.mode} pp{size}: prompt_n={u['prompt_tokens']} cache_n={out[f'pp{size}']['cache_n']} "
              f"prefill={out[f'pp{size}']['prompt_per_second']:.1f} t/s", flush=True)
    path = os.path.join(outdir, f"results-{args.mode}-pp0.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
    print(f"wrote {path}", flush=True)


def run_real(srv, args, outdir):
    # Generated text goes in its own directory: the run directory is what gets
    # copied into a report's attachments, and prompts/outputs are explicitly not
    # copied. Keeping them one level down makes the rule "top-level files only"
    # instead of a per-filename exclusion.
    resp_dir = os.path.join(outdir, "responses")
    os.makedirs(resp_dir, exist_ok=True)
    salt_base = "vsb-real-%d" % int(time.time())
    srv.complete("warmup", 8, temperature=0.0, ignore_eos=False, cache_salt=salt_base + "-warm")
    records = []
    for name, prompt in REAL_PROMPTS:
        m0 = srv.metrics()
        r = srv.complete(prompt, 1200, temperature=0.7, top_p=0.9, ignore_eos=False,
                         cache_salt=f"{salt_base}-{name}")
        m1 = srv.metrics()
        u = r["usage"]
        gen_s = r["total"] - r["ttft"]
        dn = delta(m0, m1, "spec_draft_tokens") or 0
        acc = delta(m0, m1, "spec_accepted_tokens") or 0
        rec = {
            "prompt": name,
            "prompt_chars": len(prompt),
            "n_predict": 1200, "temperature": 0.7, "top_p": 0.9,
            "ttft_ms": round(r["ttft"] * 1000.0, 1),
            "prompt_n": u["prompt_tokens"],
            "prefill_tok_s": round(u["prompt_tokens"] / r["ttft"], 3),
            "predicted_n": u["completion_tokens"],
            "decode_tok_s": round(u["completion_tokens"] / gen_s, 3) if gen_s > 0 else None,
            "draft_n": int(dn), "accepted": int(acc),
            "accept_rate": round(acc / dn, 3) if dn else None,
            "tokens_per_cycle": round(u["completion_tokens"] / dn, 3) if dn else None,
            "wall_s": round(r["total"], 1),
            "finish_reason": r["finish_reason"],
            "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
        }
        open(os.path.join(resp_dir, f"{name}.txt"), "w", encoding="utf-8").write(r["text"])
        records.append(rec)
        print(json.dumps({k: rec[k] for k in
                          ("prompt", "prompt_n", "prefill_tok_s", "predicted_n",
                           "decode_tok_s", "accept_rate")}, ensure_ascii=False), flush=True)
    json.dump(records, open(os.path.join(outdir, "results-real.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="tensor", help="series name used in results-<mode>.json")
    ap.add_argument("--devices", default="", help="label only, e.g. CUDA1,CUDA2")
    ap.add_argument("--guard-devices", default="", help="nvidia-smi indices to sample, e.g. 1,2")
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--stages", default="0,32000,64000,128000,196000,258000")
    ap.add_argument("--n-predict", type=int, default=1000)
    ap.add_argument("--pp0-sizes", default="512,2048,8192")
    ap.add_argument("--ratio-init", type=float, default=4.3)
    ap.add_argument("--machine", default="")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--no-real", action="store_true")
    # The ladder deliberately reuses each stage's prefix in the next one, so the
    # salt is constant within a run -- it only isolates one run from the next,
    # which is what makes a repeat a cold measurement instead of a cache hit.
    ap.add_argument("--cache-salt", default="",
                    help='vLLM cache_salt for the ladder; "auto" generates a per-run one')
    ap.add_argument("--api-key", default=os.environ.get("LLM_DEPTH_BENCH_API_KEY")
                    or os.environ.get("OPENAI_API_KEY", ""),
                    help="bearer token for the server; defaults to $LLM_DEPTH_BENCH_API_KEY "
                         "then $OPENAI_API_KEY")
    ap.add_argument("--engine-label", default="",
                    help="figure subtitle; defaults to what the server reports about itself")
    ap.add_argument("--argv-file", default="",
                    help="file holding the server's launch arguments; required when the "
                         "server is remote, in another container, or not on Linux")
    ap.add_argument("--gpu-sample-interval", type=float, default=2.0,
                    help="seconds between nvidia-smi samples during the run; 0 disables")
    args = ap.parse_args()

    if args.cache_salt == "auto":
        args.cache_salt = "vsb-ladder-%d-%d" % (int(time.time()), random.randrange(10 ** 6))
    outdir = os.path.join(args.runs_dir, args.tag)
    if os.path.exists(os.path.join(outdir, f"results-{args.mode}.json")):
        print(f"BENCH-ABORT: {outdir}/results-{args.mode}.json already exists - use a new tag")
        sys.exit(1)
    os.makedirs(outdir, exist_ok=True)
    guard_idx = [x.strip() for x in args.guard_devices.split(",") if x.strip()]
    srv = Server(args.url, args.model, args.api_key)

    srv.complete("ping", 4, temperature=0.0, ignore_eos=False)
    if args.cache_salt:
        srv.probe_cache_salt()
    try:
        fingerprint = srv._post("/v1/completions", {"model": args.model, "prompt": "ping",
                                                    "max_tokens": 1, "temperature": 0},
                                timeout=120).get("system_fingerprint", "")
    except Exception:
        fingerprint = ""
    snapshot = server_snapshot(srv)
    ver = snapshot.get("version", "")
    argv, argv_source = read_argv(args)
    if not argv:
        print(f"WARNING: no server launch arguments recorded - {argv_source}", flush=True)
    open(os.path.join(outdir, f"argv-{args.mode}.txt"), "w", encoding="utf-8").write(argv + "\n")

    print(f"server ok (vllm {ver}), tag={args.tag}, mode={args.mode}", flush=True)
    sampler = GpuSampler(guard_idx, args.gpu_sample_interval)
    sampler.start()
    try:
        records, err = run_ladder(srv, args, outdir, guard_idx, sampler)
        if err:
            print(f"BENCH-ABORT: {err}")
            sys.exit(1)
        run_pp0(srv, args, outdir)
        if not args.no_real:
            run_real(srv, args, outdir)
    finally:
        sampler.stop()
        csv_path = sampler.write_csv(os.path.join(outdir, f"gpu-telemetry-{args.mode}.csv"))
        if csv_path:
            print(f"wrote {csv_path} ({len(sampler.rows)} samples)", flush=True)

    deepest = max((r.get("effective_depth") or 0) for r in records)
    cache_dtype = cache_config(snapshot).get("cache_dtype", "")
    info = {
        "tag": args.tag,
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "ctx": args.ctx, "stages": args.stages, "n_predict": args.n_predict,
        "modes": [{"name": args.mode, "device": args.devices,
                   "argv_file": f"argv-{args.mode}.txt"}],
        "machine": args.machine,
        "bin": "already-running server (not launched by this tool)",
        "bin_version": f"vllm {ver}",
        "bin_sha256": "",
        "server_fingerprint": fingerprint,
        "cache_k": cache_dtype, "cache_v": cache_dtype,
        "launch_prefix": "", "tensor_split": "",
        "engine_label": args.engine_label or default_engine_label(ver, snapshot),
        "harness": "llm-depth-bench.py (llama-split-bench protocol over an OpenAI-compatible API)",
        "deepest_measured": deepest,
        "tokenize_calls": srv.tokenize_calls,
        "gpu_sample_interval_s": args.gpu_sample_interval,
        "gpu_probe_backend": _PROBE.backend,
        "token_count_backend": srv.token_backend,
        "cache_salt_mode": srv.cache_salt_mode,
        "metric_names": srv.metric_names,
        "argv_source": argv_source,
        "server_config": snapshot,
        "gpu_samples": len(sampler.rows),
    }
    json.dump(info, open(os.path.join(outdir, "run-info.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("run-info.json written", flush=True)
    print(f"BENCH-DONE {args.tag} {datetime.datetime.now().strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
