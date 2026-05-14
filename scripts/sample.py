"""Friendly Python wrapper around `infer_modern_gpt.exe` for BPE-tokenized models.

Usage:
  python scripts/sample.py \\
      --ckpt runs/modern_gpt.ckpt \\
      --meta data/tinyshakespeare_cl100k.bin.meta \\
      --prompt "Once upon a time" \\
      --num 200 --temp 0.8 \\
      --d-model 128 --n-heads 4 --n-layers 4 --d-ffn 512 --seq-len 64

The wrapper:
  1. Reads .meta to find the tiktoken encoding (cl100k_base, etc.).
  2. Tokenizes the prompt.
  3. Spawns infer_modern_gpt with `--prompt-tokens` + `--output-tokens-only`.
  4. Decodes the returned token IDs back to text.
"""

import argparse
import os
import shutil
import subprocess
import sys

import tiktoken


def read_meta(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/modern_gpt.ckpt")
    p.add_argument("--meta", required=True,
                    help="path to <tokens>.bin.meta produced by prepare_tokens.py")
    p.add_argument("--prompt", default="")
    p.add_argument("--num", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-kv-heads", type=int, default=0,
                    help="GQA n_kv_heads; 0 = same as n_heads (MHA)")
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ffn", type=int, default=512)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--infer-bin",
                    default=os.path.join("llm", "build", "infer_modern_gpt.exe"))
    args = p.parse_args()

    if not os.path.isfile(args.meta):
        print(f"meta file not found: {args.meta}", file=sys.stderr)
        return 2

    meta = read_meta(args.meta)
    encoding = meta.get("encoding", "cl100k_base")
    enc = tiktoken.get_encoding(encoding)

    prompt_text = args.prompt or "\n"
    prompt_ids = enc.encode_ordinary(prompt_text)
    if not prompt_ids:
        prompt_ids = [enc.encode_ordinary("\n")[0]]
    csv = ",".join(str(i) for i in prompt_ids)

    if not os.path.isfile(args.infer_bin):
        print(f"infer binary not found: {args.infer_bin}", file=sys.stderr)
        print(f"build it first: scripts\\build_and_test.bat", file=sys.stderr)
        return 3

    cmd = [
        args.infer_bin,
        "--ckpt", args.ckpt,
        "--meta", args.meta,
        "--prompt-tokens", csv,
        "--output-tokens-only",
        "--num", str(args.num),
        "--seq-len", str(args.seq_len),
        "--d-model", str(args.d_model),
        "--n-heads", str(args.n_heads),
        "--n-kv-heads", str(args.n_kv_heads),
        "--n-layers", str(args.n_layers),
        "--d-ffn", str(args.d_ffn),
        "--temp", str(args.temp),
        "--seed", str(args.seed),
    ]
    print(f"# encoding={encoding}  prompt_ids={len(prompt_ids)}",
           file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    out_line = proc.stdout.strip()
    if not out_line:
        print("no tokens emitted", file=sys.stderr)
        return 4
    gen_ids = [int(x) for x in out_line.split()]

    full_text = enc.decode(prompt_ids + gen_ids)
    sys.stdout.write(full_text)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
