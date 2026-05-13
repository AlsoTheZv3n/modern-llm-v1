# 03 — Data Pipeline

> All data is streamed directly from HuggingFace Hub — nothing is downloaded in bulk.

---

## Streaming Strategy

### Why Streaming

Training on 100B+ tokens requires terabytes of data. Local storage isn't viable. HuggingFace's Parquet streaming lets you fetch batches on demand over HTTP — the OS caches recent data but you never store the full dataset.

### How HuggingFace Streaming Works

HuggingFace datasets are stored as Parquet files. Each file is ~500MB–1GB. The streaming API fetches rows on demand via HTTP range requests. You get:
- Zero disk usage (OS page cache handles recency)
- Resume-safe (track shard + row offset in checkpoint)
- Multi-shard parallelism (multiple HTTP connections)

### C++ Streaming Client

Since our training loop is in C++, we need a lightweight client. Options:

**Option A — Sidecar Python process:**
- Small Python script streams from HF and writes tokenized batches to a shared memory buffer or Unix domain socket
- C++ training loop reads from that buffer
- Simplest to implement, Python handles all HF API complexity

**Option B — Pure C++ with libcurl:**
- Use libcurl to fetch Parquet shard URLs from HF Hub API
- Use Apache Arrow C++ library to decode Parquet rows
- Extract `text` field, feed into C++ BPE tokenizer
- More work but fully self-contained

**Recommended: Option A for v1** (sidecar is 50 lines of Python, unblocks training fast)

```python
# data_server.py — runs alongside C++ trainer
from datasets import load_dataset
import socket, struct, json

dataset = load_dataset("HuggingFaceFW/fineweb-edu", 
                       name="sample-10BT", 
                       split="train", 
                       streaming=True)

# Shuffle buffer
dataset = dataset.shuffle(seed=42, buffer_size=10000)

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind("/tmp/dataserver.sock")
sock.listen(1)
conn, _ = sock.accept()

for sample in dataset:
    text = sample["text"].encode("utf-8")
    length = struct.pack("I", len(text))
    conn.sendall(length + text)
```

---

## Pre-Training Datasets

### Primary — General Knowledge + Reasoning

| Dataset | HF ID | Size | Quality | Why |
|---|---|---|---|---|
| **FineWeb-Edu** | `HuggingFaceFW/fineweb-edu` | 1.3T tokens | ⭐⭐⭐⭐⭐ | Best web text, educationally filtered, no garbage |
| **FineWeb-2** | `HuggingFaceFW/fineweb-2` | Multi-lang | ⭐⭐⭐⭐⭐ | Multilingual FineWeb |
| **Dolma** | `allenai/dolma` | 3T tokens | ⭐⭐⭐⭐ | Mix of web, academic, code, books |
| **DCLM-Baseline** | `mlfoundations/dclm-baseline-1.0` | 4T tokens | ⭐⭐⭐⭐ | DataComp filtered, great quality/diversity |

### Code

| Dataset | HF ID | Size | Why |
|---|---|---|---|
| **The Stack v2** | `bigcode/the-stack-v2` | 900B tokens | Permissively licensed code, all languages |
| **StarCoder data** | `bigcode/starcoderdata` | 250B tokens | Curated, deduplicated code |

### Math + Reasoning

| Dataset | HF ID | Size | Why |
|---|---|---|---|
| **FineMath** | `HuggingFaceTB/finemath` | 34B–54B tokens | Educational math content, excellent quality |
| **OpenWebMath** | `open-web-math/open-web-math` | 15B tokens | Web math (arxiv, StackExchange, etc.) |
| **Proof-Pile-2** | `EleutherAI/proof-pile-2` | 55B tokens | Mathematical proofs, formal reasoning |

### Academic / Science

| Dataset | HF ID | Why |
|---|---|---|
| **PubMed Abstracts** | `pubmed` | Medical/scientific reasoning |
| **ArXiv** | `togethercomputer/RedPajama-Data-1T` (arxiv subset) | Scientific writing |
| **S2ORC** | `allenai/s2orc` | Semantic scholar papers |

---

## Training Mix (Pre-training)

Recommended token budget for 100B total:

| Source | Tokens | % |
|---|---|---|
| FineWeb-Edu | 60B | 60% |
| The Stack v2 (code) | 15B | 15% |
| FineMath | 10B | 10% |
| Dolma (books + academic) | 10B | 10% |
| OpenWebMath | 5B | 5% |

**Epoch strategy:** Never repeat data in pre-training. If you run out of FineWeb-Edu (unlikely at 100B since it has 1.3T), move to FineWeb-2.

---

## Quality Filtering (in C++ / Python sidecar)

Apply these filters to every sample before tokenization:

```python
def quality_filter(text: str) -> bool:
    # 1. Length filter
    if len(text) < 100 or len(text) > 100_000:
        return False
    
    # 2. Repetition filter (paragraph-level dedup)
    lines = text.split("\n")
    unique_lines = set(l.strip() for l in lines if l.strip())
    if len(unique_lines) / max(len(lines), 1) < 0.5:
        return False  # >50% duplicate lines
    
    # 3. Language filter (skip non-Latin script for v1)
    # Use langdetect or fasttext-langid
    
    # 4. Perplexity filter (optional — filter very high perplexity text)
    # Requires a small reference model — skip for v1
    
    # 5. Boilerplate detection
    boilerplate = ["cookie policy", "terms of service", 
                   "click here to", "subscribe to our newsletter"]
    if any(b in text.lower() for b in boilerplate):
        return False
    
    return True
```

FineWeb-Edu already has most of this filtering done — apply lighter filters there.

---

## Tokenization

Use a BPE tokenizer compatible with **LLaMA-3 tokenizer** (128K vocab) or train a custom 32K vocab tokenizer on a sample of the data.

**For v1:** Use the LLaMA-3 tokenizer vocab file (available on HuggingFace) — load into your C++ BPE implementation. This gives you a battle-tested tokenizer without training from scratch, and makes your model's outputs directly comparable to other models.

**Document format:**
```
<|begin_of_text|>
{document text}
<|end_of_text|>
```

Pack multiple short documents into a single context window (up to max_seq_len=8192) with `<|end_of_text|>` separating them. This maximizes GPU utilization vs padding.

---

## Checkpoint Tracking

Store data position in every checkpoint so training can resume exactly:

```json
{
  "step": 50000,
  "dataset": "fineweb-edu",
  "shard_index": 142,
  "row_offset": 88420,
  "tokens_seen": 5_000_000_000,
  "model_path": "checkpoints/step_50000.bin"
}
```

---

## Data Pipeline Architecture

```
HuggingFace Hub (HTTP/Parquet)
        ↓
Python sidecar (streaming + filtering + tokenization)
        ↓ Unix socket
C++ ring buffer (8 batches prefetched)
        ↓
C++ training loop (consumes one batch at a time)
        ↓
GPU / CPU training
```
