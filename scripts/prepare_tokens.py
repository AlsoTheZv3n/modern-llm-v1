"""Pre-tokenize a corpus to a flat int32 binary file the C++ trainer reads.

Two input modes:
  1. Local text file:  --input-text data/tinyshakespeare.txt
  2. HuggingFace data: --hf-dataset HuggingFaceFW/fineweb-edu --hf-config sample-10BT

Output:
  <output>.bin   raw int32 little-endian token IDs
  <output>.meta  key=value text — read by the C++ trainer to set vocab_size

Usage examples:
  python scripts/prepare_tokens.py \\
      --input-text data/tinyshakespeare.txt \\
      --output data/tinyshakespeare_cl100k.bin

  python scripts/prepare_tokens.py \\
      --hf-dataset HuggingFaceFW/fineweb-edu --hf-config sample-10BT \\
      --hf-text-field text --hf-split train \\
      --max-tokens 50_000_000 \\
      --output data/finewebedu_50m_cl100k.bin
"""

import argparse
import json
import os
import sys
import time
from typing import Iterable

import numpy as np
import tiktoken


def stream_chunks_from_text_file(path: str, chunk_chars: int = 1_000_000) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                return
            yield chunk


def stream_text_from_hf(name: str, config: str, split: str, text_field: str) -> Iterable[str]:
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, streaming=True)
    for ex in ds:
        text = ex.get(text_field)
        if text:
            yield text


def encode_stream_to_file(enc: tiktoken.Encoding, chunks: Iterable[str],
                            out_path: str, max_tokens: int,
                            batch_docs: int = 256, num_threads: int = 8,
                            log_every: int = 5_000_000) -> int:
    """Stream chunks → tiktoken (batched, multi-threaded) → int32 binary file.
    RAM stays bounded by one batch's worth of tokens (~MB), so this scales
    to 10B+ tokens without materializing the full array.
    Returns the number of tokens written."""
    eot = enc.eot_token  # may be None
    total = 0
    last_log = 0
    started = time.time()

    def flush_batch(batch: list[str], fout, current_total: int) -> tuple[int, bool]:
        """Encode batch with tiktoken's parallel API, append eot per doc,
        write as int32 to fout. Returns (new_total, hit_cap)."""
        if not batch:
            return current_total, False
        encoded = enc.encode_ordinary_batch(batch, num_threads=num_threads)
        for ids in encoded:
            if eot is not None:
                ids.append(eot)
            arr = np.asarray(ids, dtype=np.int32)
            if max_tokens and current_total + len(arr) >= max_tokens:
                keep = max_tokens - current_total
                arr = arr[:keep]
                fout.write(arr.tobytes())
                current_total += len(arr)
                return current_total, True
            fout.write(arr.tobytes())
            current_total += len(arr)
        return current_total, False

    with open(out_path, "wb", buffering=4 * 1024 * 1024) as fout:
        batch: list[str] = []
        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= batch_docs:
                total, hit_cap = flush_batch(batch, fout, total)
                batch = []
                if hit_cap:
                    break
                if total - last_log >= log_every:
                    elapsed = max(time.time() - started, 1e-6)
                    rate = total / elapsed
                    eta = ""
                    if max_tokens:
                        remaining = max_tokens - total
                        secs = remaining / max(rate, 1.0)
                        eta = f"  eta {secs/60:5.1f} min"
                    print(f"  ... {total:>13,} / "
                          f"{max_tokens if max_tokens else 0:>13,} tokens"
                          f"  ({rate/1e3:.0f} k/s){eta}",
                          flush=True, file=sys.stderr)
                    last_log = total
        # Final partial batch
        total, _ = flush_batch(batch, fout, total)

    return total


def main() -> int:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-text", help="local UTF-8 text file")
    src.add_argument("--hf-dataset", help="HuggingFace dataset name (streaming)")
    p.add_argument("--hf-config", default=None,
                    help="dataset config (e.g., sample-10BT for fineweb-edu)")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--hf-text-field", default="text")
    p.add_argument("--encoding", default="cl100k_base",
                    choices=["r50k_base", "p50k_base", "cl100k_base", "o200k_base"])
    p.add_argument("--output", required=True, help="output .bin path")
    p.add_argument("--max-tokens", type=int, default=0,
                    help="hard cap on number of tokens to emit (0 = no cap)")
    args = p.parse_args()

    enc = tiktoken.get_encoding(args.encoding)
    vocab_size = enc.n_vocab
    print(f"encoding: {args.encoding}  (vocab_size = {vocab_size:,})",
          file=sys.stderr)

    if args.input_text:
        if not os.path.isfile(args.input_text):
            print(f"error: not a file: {args.input_text}", file=sys.stderr)
            return 1
        print(f"source: text file {args.input_text}", file=sys.stderr)
        chunks = stream_chunks_from_text_file(args.input_text)
        source_id = args.input_text
    else:
        cfg_str = f"/{args.hf_config}" if args.hf_config else ""
        print(f"source: hf dataset {args.hf_dataset}{cfg_str} "
              f"(split={args.hf_split}, text-field={args.hf_text_field})",
              file=sys.stderr)
        chunks = stream_text_from_hf(args.hf_dataset, args.hf_config,
                                       args.hf_split, args.hf_text_field)
        source_id = (f"hf://{args.hf_dataset}{cfg_str}"
                      f"/{args.hf_split}#{args.hf_text_field}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    started = time.time()
    n_tokens = encode_stream_to_file(enc, chunks, args.output, args.max_tokens)
    elapsed = time.time() - started

    if n_tokens == 0:
        print("error: produced 0 tokens", file=sys.stderr)
        return 2

    nbytes = n_tokens * 4
    print(f"wrote {n_tokens:,} int32 tokens "
          f"({nbytes / 1024 / 1024:.1f} MiB) to {args.output} "
          f"in {elapsed:.1f}s "
          f"({n_tokens / max(elapsed, 1e-6) / 1000:.0f} k tokens/s)",
          file=sys.stderr)

    meta_path = args.output + ".meta"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"encoding={args.encoding}\n")
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"num_tokens={n_tokens}\n")
        f.write(f"source={source_id}\n")
        f.write(f"max_tokens={args.max_tokens}\n")
    print(f"wrote {meta_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
