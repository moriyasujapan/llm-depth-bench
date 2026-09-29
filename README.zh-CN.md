# llm-depth-bench

针对**已经在运行的** OpenAI 兼容 LLM 服务器，测量 **prefill 与 decode 吞吐随上下文
深度的变化**。适用于不是你启动的、无法重启、甚至不在本机上的服务器。

**English → [README.md](README.md) ・ 日本語 → [README.ja.md](README.ja.md)**

本工具重新实现了
[llama-split-bench](https://github.com/kuraneko1/llama-split-bench)（kuraneko1，MIT）
的测量协议与输出结构，因此产生的 JSON 可以直接交给该项目的 `plot_bench.py` 绘图，
数值含义也与用该工具产出的报告一致。

## 为什么需要它

llama-split-bench 会自己启动 llama.cpp 的 `llama-server`，并读取 `/completion`
返回的 `timings` 字段。当测量目标是一个长期运行的 vLLM 部署时，这两个前提都不成立：
服务器已经在跑（往往还在为别人提供服务），而 OpenAI 兼容 API 并没有 `timings`。

因此本工具从一次流式补全中，在客户端侧推导出同样的量：

| llama.cpp 的 `timings` | 本工具的求法 |
|---|---|
| `prompt_per_second` | （本次新评估的 prompt token 数）/ TTFT |
| `predicted_per_second` | `completion_tokens` /（总时间 − TTFT） |
| `cache_n` | `/metrics` 中 `prefix_cache_hits_total` 的差值 |
| `draft_n` / `draft_n_accepted` | `/metrics` 中 `spec_decode_*` 的差值 |

TTFT 是流中第一个 token 到达为止的墙钟时间，因此 prefill 数值**包含网络与调度开销**。
在本地回环、服务器空闲的情况下这部分很小，但不为零。这也是为什么本工具给出的是
**「按当前部署状态的服务器性能」**，而不是「GPU 本身的性能」。

## 测量内容

1. **深度阶梯（复用上下文）** — 一条逐级增长的 prompt（0 → 32k → … → 目标）。
   每一级记录*增量*的 prefill 速度，随后生成 N 个 token（`ignore_eos`，默认 1000）
   得到 decode 速度。这正是真实 agent 使用的形态：很长的 prompt，然后在很深的位置
   开始生成。
2. **深度 0 的 prefill** — 用 512 / 2048 / 8192 token 的全新 prompt，并隔离前缀
   缓存，得到空上下文下正确的 prefill 速度。阶梯的第一级是 11 个 token 的 prompt，
   **它的 prefill 数值是测量假象，绝不能画进图里**。
3. **投机解码接受率** — 存在 draft 模型或 MTP head 时，decode 速度取决于被接受的
   draft token 数量。合成阶梯文本的接受率会高到 0.9 以上，使 decode 看起来过于乐观。
4. **真实 prompt 修正** — 另外用 3 条代表实际负载的 prompt 以 temperature 0.7 运行，
   得出修正系数（**务必阅读下面的注意事项**）。
5. **运行凭证** — `run-info.json` 记录服务器自述的配置、启动参数的获取途径、
   整个运行期间采集的 GPU 遥测，以及从 metrics 还原出的投机解码结构。

## 环境要求

- Python 3.8 以上 — **仅标准库**，无需 pip install。
- 一个可访问的、支持流式 `/v1/completions` 的 OpenAI 兼容服务器。
- 仅绘图时需要：`matplotlib` 和 llama-split-bench 的 `plot_bench.py`。
- 仅采集 GPU 遥测时需要：NVIDIA 驱动（直接调用 NVML，见下文）。

## 快速开始

```bash
git clone https://github.com/moriyasujapan/llm-depth-bench && cd llm-depth-bench
mkdir -p runs

./llm-depth-bench.py \
  --tag my-first-run \
  --url http://127.0.0.1:8000 \
  --model 实际 served 的模型名 \
  --mode tp2 \
  --devices CUDA0,CUDA1 \
  --guard-devices 0,1 \
  --ctx 262144 \
  --stages 0,32000,64000,128000,196000,258000 \
  --n-predict 1000 \
  --cache-salt auto \
  --machine "不含主机名的中性机器描述"
```

如果服务器需要密钥，请传入 `--api-key`，或在环境中设置
`LLM_DEPTH_BENCH_API_KEY`（或 `OPENAI_API_KEY`）。

深阶梯需要数十分钟。请以后台分离方式启动并轮询日志：成功时打印
`BENCH-DONE <tag> <时间>`，失败时打印 `BENCH-ABORT: …`，因此不需要交互式终端
（不要用 `tail -f`，它不会结束）。

```bash
setsid nohup ./llm-depth-bench.py --tag my-first-run … > runs/my-first-run.log 2>&1 &
grep -E 'BENCH-(DONE|ABORT)' runs/my-first-run.log
```

按深度测量「接近真实使用」的 decode（把深度的影响与接受率的影响分开）：

```bash
./depth_real.py --url http://127.0.0.1:8000 --model 模型名 \
  --stages 0,32000,128000,258000 --out runs/my-first-run/results-real-depth.json
```

## 绘图

图由 llama-split-bench 自带的 `plot_bench.py` 生成，因此本工具产出的报告与该工具
产出的报告外观一致：

```bash
python plot_bench.py --dir runs/my-first-run --series tp2 --lang en --estimate off
```

`plot_bench.py` 把副标题硬编码成了 "llama.cpp"，本仓库附带一行补丁
[`plot_bench-engine_label.patch`](plot_bench-engine_label.patch)，让它优先使用
`run-info.json` 中的 `engine_label` 字段。传入 `--series a,b` 可以把两次运行画在
一起，**比较两种服务器配置**就是这样做的。

## 哪些部分与引擎无关，哪些不是

计时的核心只需要流式 `/v1/completions`，因此任何 OpenAI 兼容服务器都能跑。而所有
提升测量精度的部分目前都依赖 vLLM：

| 功能 | 优先使用 | 缺失时的回退 |
|---|---|---|
| 计时（prefill / decode） | `POST /v1/completions`（流式） | — 必需，到处都能用 |
| 认证 | `--api-key`、`$LLM_DEPTH_BENCH_API_KEY`、`$OPENAI_API_KEY` | 设置后以 `Authorization: Bearer` 发送 |
| prompt 定长 | `POST /tokenize`（`count`，llama.cpp 用 `len(tokens)`） | **自动**：用 `max_tokens=1` 的补全读取 `usage.prompt_tokens`。到处可用，但每次探测都是一次真实 prefill |
| 缓存命中的剥离 | `prefix_cache_hits_total` 类指标 | 该列缺失；prefill 按整条 prompt 计算，**命中缓存的部分会让数值虚高** |
| 接受率 | `spec_*_draft_tokens_total` / `..._accepted_tokens_total` | 该列缺失 |
| 争用检查 | `num_requests_running` / `num_running_reqs` 类指标 | 有其他请求在跑时不会警告 |
| 每次运行的缓存隔离 | `cache_salt` 请求字段 | **自动**：改为把 salt 加在 prompt 前面 |
| GPU 遥测 | NVML | `nvidia-smi`，再没有则不记录 |

指标名通过「去掉引擎前缀后的名字」做模式匹配来解析，因此只要某引擎用自己的前缀
发布了同一概念，无需改代码即可识别。**不做任何猜测**：每个逻辑指标最终绑定到了
哪个指标名，都会写入 `run-info.json` 的 `metric_names`；匹配不到的概念直接缺失。
**在相信任何派生列之前，请先核对这个绑定** —— 开发过程中正是靠它发现了误绑定到
「按位置拆分」指标的问题。

**已验证环境：** Linux + NVIDIA GPU 上的 vLLM `0.1.dev20073`（全功能）；以及一个
没有 `/tokenize`、`/metrics`、`/version` 的带认证 OpenAI 兼容代理（确认计时列正常
产出，其余优雅缺失）。**llama.cpp 的 `llama-server` 与 SGLang 尚未试过**，回退逻辑
已写好但未经验证。

## 相信任何数字之前，请先知道这些坑

- **只要打算重复运行，就必须加 `--cache-salt auto`。** 否则第二次跑同一条阶梯会
  命中前缀缓存：32k 那一级实际只评估两三千个新 token，而不是 32,000 个，却仍然
  会给出一个「看起来像 prefill 速度」的数值，**你根本察觉不到自己量的是另一回事**。
  这是用本工具骗过自己的最简单方式。
- **绝不要把阶梯深度 0 的 prefill 画进图里。** 那是一条 11 个 token 的 prompt。
- **对启用了投机解码的模型，真实 prompt 修正系数并不可靠。** llama-split-bench
  用（真实 prompt 的 decode）÷（阶梯深度 0 的 decode）计算，但深度 0 太短，draft
  还没热起来，系数可能大于 1。遇到这种情况请用 `--estimate off` 出图。
- **decode 取决于「让它生成什么」，而不只是「有多深」。** 在某个 MTP 模型上，同一
  深度下合成阶梯文本得到 178 tok/s（接受率 0.93），而接近真实使用的生成只有
  70 tok/s（接受率 0.21）。**报告 decode 时请务必同时给出接受率。**
- **`results-real-depth.json` 每个深度只有一个样本，且 temperature 为 0.7。**
  请按 ±30% 的波动来看待。
- **测量共享服务器可行，但并非没有代价。** 每一级开始前会检查
  `num_requests_running` 并给出警告，但不会中断。
- **服务器的连续运行时长也是一个变量。** 同一硬件上，重启后几分钟与几天后的两次
  运行不能想当然地拿来比较。
- **会静默丢弃未知字段的代理会让缓存隔离失效。** `cache_salt` 的探测只能看到请求
  是否被接受，因此一个配置为丢弃未知参数的网关会记录成 `cache_salt_mode: field`，
  而 salt 其实从未到达引擎。如果重复跑阶梯时 `prompt_n` 小得异常，就说明 salt 没送到。

## 输出

全部写入 `runs/<tag>/`。顶层文件可以安全公开；生成的文本放在下一层的 `responses/`
里，因此可以**按目录而不是按文件名**排除。

| 文件 | 内容 |
|---|---|
| `results-<mode>.json` | 阶梯，每一级一条记录 |
| `results-<mode>-pp0.json` | 空上下文下的 prefill |
| `results-real.json` | 真实 prompt 修正 |
| `results-real-depth.json` | 按深度的真实 decode（`depth_real.py`） |
| `run-info.json` | 运行凭证（见下） |
| `argv-<mode>.txt` | 服务器启动参数（若能取到） |
| `gpu-telemetry-<mode>.csv` | 运行期间全部 GPU 采样点 |
| `responses/*.txt` | 生成的文本，**不适合公开** |

除 llama-split-bench 原有字段外，`run-info.json` 还包含：

- `server_config` — 通过 API 读到的、引擎**实际解析后**的配置：版本、模型与
  `max_model_len`、全部 Prometheus `*_info` 指标（vLLM 上即整套 KV cache 配置），
  以及投机解码结构。**这比命令行更可信**：命令行只记录了「打算怎么配」，而环境
  变量与默认值解析都可能让实际值不同。
- `server_config.speculative` — 用两种独立方式还原的 draft 长度（draft token 数 ÷
  draft 次数，以及 `position` 标签的个数），外加**每个 draft 位置的接受占比**。
- `argv_source` — 启动参数从哪里取得，或为何缺失：Linux / macOS 用 `ps`，Windows 用
  `Get-CimInstance Win32_Process`，`/proc` 作为最后手段，远程服务器则用 `--argv-file`。
- `metric_names` —— 每个逻辑指标实际绑定到了哪个指标名。
- `token_count_backend` —— `endpoint` 或 `usage`（见上表）。
- `cache_salt_mode` —— `field` 或 `prefix`。
- `gpu_probe_backend`、`gpu_samples`、`tokenize_calls`。

每一级的记录中含 `gpu_window`：该级时间窗内功耗、SM / 显存频率、利用率与温度的
min / mean / max。注意时间窗从请求发出前就开始，**最小值可能是一个空闲采样**；
要回答「是否撞到功耗墙」请看最大值。

## GPU 遥测

采样通过 `ctypes` 直接调用 NVML（`libnvidia-ml.so.1`，Windows 上为 `nvml.dll`），
在无法加载时回退到解析 `nvidia-smi`。两者数值完全一致（`nvidia-smi` 本身就是 NVML
的客户端），但 **NVML 每次采样约 0.75 ms，而 fork 一次 `nvidia-smi` 约 61 ms**，
相差 80 倍以上。因此用 NVML 时即使每秒采样一次，也不会让「测量本身」成为被测对象的
一部分。用 `--gpu-sample-interval 0` 可关闭。

## 平台支持

端到端实跑过的只有 Linux。代码中没有 Linux 专有的部分：NVML 在 Windows 上同样可以
加载；进程发现在 Linux / macOS 用 `ps`，在 Windows 用 CIM 查询（**刻意不使用
`wmic`**，它已从现行 Windows 中移除）；所有文件写入都显式指定 UTF-8。欢迎反馈在
其他平台上的成功或失败。

## 致谢

测量协议、输出结构与图表均来自
[llama-split-bench](https://github.com/kuraneko1/llama-split-bench)（kuraneko1，MIT）。
**如果你能自己启动服务器，那个工具才是正确的选择**：它能端到端回答「layer 切分还是
tensor 切分更快」，而本工具刻意不涉足这个问题。

## 许可证

MIT — 见 [LICENSE](LICENSE)。
