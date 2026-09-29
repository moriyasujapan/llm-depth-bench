#!/usr/bin/env python3
"""補足計測: 合成ラダーと同じ深さで、生成内容だけを「実運用相当」に替えて decode を測る。

ラダーの TAIL（数列の続きを書け）は投機デコードの採択率が 0.9 台まで上がり、
decode が楽観側に出る。同じ前置き（キャッシュ済みの filler）に別の指示を付けて
temp 0.7 で生成することで、深さの効果と採択率の効果を分離する。

Supplementary measurement: same depths as the synthetic ladder, but with the
generated content swapped for something closer to real use. The ladder's tail
("continue the sequence") pushes speculative acceptance into the 0.9s, which
flatters decode; re-running the same cached prefix with a different instruction
at temperature 0.7 separates the depth effect from the acceptance effect.
"""
import argparse
import importlib.util
import json
import os

DEFAULT_INSTRUCTION = (
    "\n\n上の数列は完全に無視してください。まったく別の話題として、GPU の HBM 帯域幅と "
    "4bit 量子化が推論のスループットに与える影響について、あなた自身の言葉で、"
    "具体例を挙げながら詳しく論じてください。箇条書きではなく文章で書いてください。")


def load_harness():
    """Import the main harness as a module, by path relative to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "llm-depth-bench.py")
    spec = importlib.util.spec_from_file_location("vsb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default=os.environ.get("LLM_DEPTH_BENCH_API_KEY")
                    or os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="where to write results-real-depth.json")
    ap.add_argument("--stages", default="0,32000,128000,258000")
    ap.add_argument("--n-predict", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--ratio-init", type=float, default=4.3)
    ap.add_argument("--instruction-file",
                    help="file holding the instruction appended at each depth; "
                         "defaults to a built-in Japanese prompt")
    args = ap.parse_args()

    vsb = load_harness()
    instruction = DEFAULT_INSTRUCTION
    if args.instruction_file:
        with open(args.instruction_file, encoding="utf-8") as fh:
            instruction = "\n\n" + fh.read().strip()

    srv = vsb.Server(args.url, args.model, args.api_key)
    out = []
    ratio = args.ratio_init
    pcache = {}
    for target in [int(x) for x in args.stages.split(",")]:
        if target == 0:
            prompt = instruction.strip()
        else:
            body, _, ratio = vsb.sized_prompt(srv, target, ratio, pcache)
            prompt = body[:-len(vsb.TAIL)] + instruction
        m0 = srv.metrics()
        r = srv.complete(prompt, args.n_predict, temperature=args.temperature,
                         top_p=args.top_p, ignore_eos=True)
        m1 = srv.metrics()
        u = r["usage"]
        gen = r["total"] - r["ttft"]
        hits = int(vsb.delta(m0, m1, "prefix_cache_hits") or 0)
        dn = int(vsb.delta(m0, m1, "spec_draft_tokens") or 0)
        acc = int(vsb.delta(m0, m1, "spec_accepted_tokens") or 0)
        rec = {"target_tokens": target, "effective_depth": u["prompt_tokens"],
               "cache_n": hits, "prompt_n": u["prompt_tokens"] - hits,
               "prompt_ms": round(r["ttft"] * 1000, 1),
               "prompt_per_second": (round((u["prompt_tokens"] - hits) / r["ttft"], 2)
                                     if r["ttft"] > 0 else None),
               "predicted_n": u["completion_tokens"],
               "predicted_per_second": round(u["completion_tokens"] / gen, 2) if gen > 0 else None,
               "draft_n": dn, "draft_n_accepted": acc,
               "draft_accept_rate": round(acc / dn, 3) if dn else None,
               "temperature": args.temperature, "top_p": args.top_p,
               "wall_s": round(r["total"], 1)}
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
