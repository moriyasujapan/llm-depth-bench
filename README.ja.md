# llm-depth-bench

**すでに稼働している** OpenAI 互換 LLM サーバに対して、**コンテキスト深度に対する
prefill / decode スループット**を測るツールです。自分で起動していない、再起動でき
ない、あるいは同じマシンにすらないサーバを測るために作りました。

**English → [README.md](README.md) ・ 简体中文 → [README.zh-CN.md](README.zh-CN.md)**

計測プロトコルと出力スキーマは
[llama-split-bench](https://github.com/kuraneko1/llama-split-bench)（kuraneko1、MIT）
の再実装です。出力 JSON は同プロジェクトの `plot_bench.py` がそのまま読めるので、
同ツールで作られたレポートと図の形式・数値の意味が揃います。

## なぜ作ったか

llama-split-bench は llama.cpp の `llama-server` を自分で起動し、`/completion` が
返す `timings` を読みます。常駐している vLLM を測りたいとき、この前提はどちらも
成立しません。サーバはすでに動いていて（しばしば他の人が使っていて）、OpenAI 互換
API に `timings` はありません。

そこで、ストリーミング補完 1 回からクライアント側で同じ量を導出します。

| llama.cpp の `timings` | ここでの求め方 |
|---|---|
| `prompt_per_second` | （今回新規に評価されたプロンプトトークン）/ TTFT |
| `predicted_per_second` | `completion_tokens` /（総時間 − TTFT） |
| `cache_n` | `/metrics` の `prefix_cache_hits_total` の差分 |
| `draft_n` / `draft_n_accepted` | `/metrics` の `spec_decode_*` の差分 |

TTFT はストリームの最初のトークンが届くまでの実時間なので、prefill の値には
ネットワークとスケジューリングのオーバヘッドが含まれます。ループバック接続で
アイドルのサーバなら小さいものの、ゼロではありません。本ツールの数値が
「GPU の性能」ではなく**「デプロイされた状態のサーバの性能」**である理由です。

## 何を測るか

1. **深度ラダー（コンテキスト再利用）** — 段階的に伸びる 1 本のプロンプト
   （0 → 32k → … → 目標）。各段で*増分*の prefill 速度を記録し、続けて N トークン
   （`ignore_eos`、既定 1000）生成した速度が decode です。長いプロンプトを入れて
   深い位置で生成する、実際のエージェント利用の形を模しています。
2. **深さ 0 の prefill** — 512 / 2048 / 8192 トークンの新規プロンプトを、prefix
   キャッシュを隔離して送り、空コンテキストでの正しい prefill 速度を測ります。
   ラダーの初段は 11 トークンのプロンプトなので、**その prefill 値は計測上の
   アーティファクトであり、図に載せてはいけません**。
3. **投機デコードの採択率** — ドラフトモデルや MTP ヘッドがあると、decode 速度は
   ドラフトが何個採択されたかに依存します。合成ラダーのテキストは採択率が 0.9 台
   まで上がるため、decode が大きく楽観側に出ます。
4. **実プロンプト補正** — ワークロード代理のプロンプト 3 本を temperature 0.7 で
   別途実行し、補正係数を出します（後述の注意を必ず読んでください）。
5. **実行証跡** — `run-info.json` に、サーバ自身が申告した設定、起動引数の入手
   経路、実行中ずっと取った GPU テレメトリ、メトリクスから復元した投機デコードの
   段数を記録します。

## 必要なもの

- Python 3.8 以降 — **標準ライブラリのみ**。pip install は不要です。
- ストリーミング `/v1/completions` を持つ OpenAI 互換サーバ。
- 図を作る場合のみ: `matplotlib` と llama-split-bench の `plot_bench.py`。
- GPU テレメトリを取る場合のみ: NVIDIA ドライバ（NVML を直接使います。後述）。

## クイックスタート

```bash
git clone https://github.com/moriyasujapan/llm-depth-bench && cd llm-depth-bench
mkdir -p runs

./llm-depth-bench.py \
  --tag my-first-run \
  --url http://127.0.0.1:8000 \
  --model 実際に served されているモデル名 \
  --mode tp2 \
  --devices CUDA0,CUDA1 \
  --guard-devices 0,1 \
  --ctx 262144 \
  --stages 0,32000,64000,128000,196000,258000 \
  --n-predict 1000 \
  --cache-salt auto \
  --machine "ホスト名を含まない中立なマシンの説明"
```

サーバがキーを要求する場合は `--api-key` を渡すか、環境変数
`LLM_DEPTH_BENCH_API_KEY`（または `OPENAI_API_KEY`）を設定してください。

深いラダーは数十分かかります。デタッチして起動し、ログをポーリングしてください。
成功時に `BENCH-DONE <tag> <時刻>`、失敗時に `BENCH-ABORT: …` を出すので、対話端末は
不要です（`tail -f` は終わらないので使わないこと）。

```bash
setsid nohup ./llm-depth-bench.py --tag my-first-run … > runs/my-first-run.log 2>&1 &
grep -E 'BENCH-(DONE|ABORT)' runs/my-first-run.log
```

深さ別に「実運用相当の生成」で decode を測る（深さの効果と採択率の効果を分離する）:

```bash
./depth_real.py --url http://127.0.0.1:8000 --model モデル名 \
  --stages 0,32000,128000,258000 --out runs/my-first-run/results-real-depth.json
```

## 図

図は llama-split-bench の `plot_bench.py` をそのまま使います。本ツールで作った
レポートが、同ツールで作ったものと同じ見た目になります。

```bash
python plot_bench.py --dir runs/my-first-run --series tp2 --lang ja --estimate off
```

`plot_bench.py` は副題に "llama.cpp" を直書きしているので、`run-info.json` の
`engine_label` を優先させる 1 行パッチ
[`plot_bench-engine_label.patch`](plot_bench-engine_label.patch) を同梱しています。
`--series a,b` と指定すると 2 つの実行を重ねて描けるので、**サーバ構成同士の比較**は
この形で行います。

## エンジン非依存な部分と、そうでない部分

計測の核はストリーミング `/v1/completions` しか使わないので、OpenAI 互換であれば
どれでも動きます。一方、計測の精度を上げている部分は現状 vLLM 固有です。

| 機能 | 優先して使うもの | 無い場合のフォールバック |
|---|---|---|
| 時間計測（prefill / decode） | `POST /v1/completions`（stream） | — 必須。どこでも動く |
| 認証 | `--api-key`、`$LLM_DEPTH_BENCH_API_KEY`、`$OPENAI_API_KEY` | 指定時は `Authorization: Bearer` で送る |
| プロンプトのサイジング | `POST /tokenize`（`count`、llama.cpp なら `len(tokens)`） | **自動**: `max_tokens=1` の補完から `usage.prompt_tokens` を読む。どこでも動くが、1 回の測定が実際の prefill になる |
| キャッシュ分の切り分け | `prefix_cache_hits_total` 相当のメトリクス | 列が出ない。prefill を総プロンプト基準で出すため、**キャッシュに当たった分だけ過大**になる |
| 採択率 | `spec_*_draft_tokens_total` / `..._accepted_tokens_total` | 列が出ない |
| 競合チェック | `num_requests_running` / `num_running_reqs` 相当 | 他リクエストが走っていても警告が出ない |
| run 単位のキャッシュ隔離 | `cache_salt` リクエストフィールド | **自動**: salt をプロンプト先頭に付ける方式に切り替わる |
| GPU テレメトリ | NVML | `nvidia-smi`、それも無ければ記録なし |

メトリクス名は接頭辞を外した名前に対するパターンで解決するので、同じ概念を自前の
接頭辞で公開しているエンジンならコード変更なしに拾えます。**推測はしません**。
論理メトリクスが何に束縛されたかは `run-info.json` の `metric_names` に書き出され、
一致しなかった概念は単に欠落します。**派生列を信じる前にこの束縛を確認してください** —
開発中に「位置別の内訳」への誤束縛を見つけたのがまさにこれです。

**動作確認済み:** Linux + NVIDIA GPU 上の vLLM `0.1.dev20073`（全機能）。および
`/tokenize`・`/metrics`・`/version` を持たない認証付き OpenAI 互換プロキシ
（時間計測の列は出て、他は欠落に degrade することを確認）。**llama.cpp の
`llama-server` と SGLang は未試行**で、フォールバックは書いてありますが未検証です。

## 数値を信じる前に知っておくべき落とし穴

- **再実行するつもりなら必ず `--cache-salt auto` を付けてください。** 付けないと
  2 回目のラダーが prefix キャッシュに当たり、32k 段が 32,000 ではなく数千トークン
  しか評価しません。それでも「prefill 速度らしい値」が出てしまうため、**まったく
  別の量を測っていることに気づけません**。このツールで自分を騙す一番簡単な方法です。
- **ラダーの深さ 0 の prefill は図に載せないこと。** 11 トークンのプロンプトです。
- **実プロンプト補正係数は投機デコード有効なモデルでは当てになりません。**
  llama-split-bench は（実プロンプト decode）÷（ラダー深さ 0 の decode）で出しますが、
  深さ 0 はドラフトが温まらないほど短いため、係数が 1 を超えることがあります。
  その場合は `--estimate off` で図を作ってください。
- **decode は「どれだけ深いか」より「何を生成させたか」で変わります。** ある MTP
  モデルでは同じ深さで、合成ラダーのテキストが 178 tok/s（採択率 0.93）、実運用
  相当の生成が 70 tok/s（採択率 0.21）でした。**decode の数値には必ず採択率を併記**
  してください。
- **`results-real-depth.json` は各深さ 1 サンプル、temperature 0.7 です。** ±30%
  程度のばらつきがあるものとして読んでください。
- **共有サーバの計測は可能ですが無料ではありません。** 各段の直前に
  `num_requests_running` を確認して警告を出しますが、中断はしません。
- **サーバの連続稼働時間も変数です。** 同じハードでも、再起動直後と数日後の実行を
  無条件に比較はできません。
- **未知のフィールドを黙って捨てるプロキシはキャッシュ隔離を無効化します。**
  `cache_salt` の判定はリクエストが受理されたかどうかしか見ていないため、知らない
  パラメータを破棄する設定のゲートウェイでは `cache_salt_mode: field` と記録され
  ながら salt がエンジンまで届きません。ラダーの再実行で `prompt_n` が不自然に
  小さければ、salt が届いていない証拠です。

## 出力

すべて `runs/<tag>/` に出ます。トップレベルのファイルは公開して差し支えないもの、
生成テキストは `responses/` に 1 段下げてあるので、**ファイル名ではなくディレクトリ
単位で除外**できます。

| ファイル | 内容 |
|---|---|
| `results-<mode>.json` | ラダー。段ごとに 1 レコード |
| `results-<mode>-pp0.json` | 空コンテキストでの prefill |
| `results-real.json` | 実プロンプト補正 |
| `results-real-depth.json` | 深さ別の実運用相当 decode（`depth_real.py`） |
| `run-info.json` | 実行証跡（下記） |
| `argv-<mode>.txt` | サーバの起動引数（取得できた場合） |
| `gpu-telemetry-<mode>.csv` | 実行中の GPU サンプル全点 |
| `responses/*.txt` | 生成テキスト。**公開向けではありません** |

`run-info.json` には llama-split-bench のフィールドに加えて次が入ります。

- `server_config` — サーバが API 経由で申告した**解決済みの**設定。バージョン、
  モデルと `max_model_len`、Prometheus の `*_info` メトリクス全部（vLLM なら KV
  キャッシュ設定一式）、投機デコードの形。**コマンドラインより信頼できます**。
  コマンドラインは「こう指定したつもり」でしかなく、環境変数や既定値の解決で
  実際の値とずれるからです。
- `server_config.speculative` — ドラフト段数を独立な 2 通り（ドラフトトークン数 ÷
  ドラフト回数、`position` ラベルの個数）で復元した値と、**位置別の採択比率**。
- `argv_source` — 起動引数をどう取ったか、取れなかったならその理由。Linux / macOS
  は `ps`、Windows は `Get-CimInstance Win32_Process`、最後の手段として `/proc`、
  リモートサーバなら `--argv-file`。
- `metric_names` — 各論理メトリクスが実際にどのメトリクス名に束縛されたか。
- `token_count_backend` — `endpoint` か `usage`（上の表を参照）。
- `cache_salt_mode` — `field` か `prefix`。
- `gpu_probe_backend`、`gpu_samples`、`tokenize_calls`。

段ごとのレコードには `gpu_window` が入り、その段の時間窓における消費電力・SM /
メモリクロック・利用率・温度の min / mean / max が記録されます。窓はリクエスト
直前から始まるので**最小値はアイドルのサンプルかもしれません**。「電力上限に
張り付いたか」を見るなら最大値を読んでください。

## GPU テレメトリ

サンプリングは `ctypes` で NVML（`libnvidia-ml.so.1`、Windows は `nvml.dll`）を
直接叩き、読めない環境では `nvidia-smi` のパースに落ちます。両者の値は同一です
（`nvidia-smi` 自体が NVML のクライアントです）が、**NVML が 1 サンプル約 0.75 ms、
`nvidia-smi` の fork が約 61 ms** と 80 倍以上違うため、NVML なら 1 秒間隔でも
「計測自体が計測対象を乱す」ことになりません。`--gpu-sample-interval 0` で無効化
できます。

## 対応プラットフォーム

E2E で実走したのは Linux のみです。コードに Linux 専用の部分はありません。NVML は
Windows でも読め、プロセス探索は Linux / macOS が `ps`、Windows が CIM クエリ
（`wmic` は現行 Windows から削除済みなので**意図的に使っていません**）、ファイル
書き込みはすべて明示的に UTF-8 です。他環境での動作報告・不具合報告を歓迎します。

## クレジット

計測プロトコル・出力スキーマ・図は
[llama-split-bench](https://github.com/kuraneko1/llama-split-bench)（kuraneko1、MIT）
のものです。**サーバを自分で起動できるなら、そちらが正しい道具です。**
「layer 分割と tensor 分割のどちらが速いか」に端から端まで答えてくれます。本ツールは
そこには踏み込みません。

## ライセンス

MIT — [LICENSE](LICENSE) を参照してください。
