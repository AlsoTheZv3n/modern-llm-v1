# modern-llm-v1

A modern decoder-only transformer LLM, **built from scratch in C++/CUDA** — no PyTorch, no abstractions. Trained on FineWeb-Edu. Monitored by a C# WPF GUI. Targets a single RTX 3080 Ti (Ampere, 12 GB VRAM).

Architecturally on par with small open-source frontier models (Olmo-2 / Qwen3-1.5 niveau): RoPE, RMSNorm, SwiGLU FFN, GQA, QK-Norm, tied embeddings, BF16 mixed precision, gradient checkpointing.

---

## Status

| Component | Status |
|---|---|
| Tensor / GEMM / AdamW / loss / RoPE / RMSNorm / SwiGLU | ✅ done |
| Multi-head attention (split/merge/repeat-KV) | ✅ done |
| GQA (Grouped-Query Attention) | ✅ done |
| QK-Norm (Olmo-2 / DeepSeek style) | ✅ done |
| Tied input/output embeddings | ✅ done |
| BF16 mixed precision (cuBLAS BF16 matmuls + BF16 mirrors + scratch arena) | ✅ done |
| BF16 optimizer states (AdamW m/v BF16, 2× memory win) | ✅ done |
| Gradient checkpointing (always-on, ~5× activation memory savings) | ✅ done |
| Flash-Attention forward (per-row, O(N) memory, correct) | ✅ done |
| Flash-Attention backward (dQ + dKdV passes, O(N) memory, correct) | ✅ done |
| BPE tokenization (cl100k_base via tiktoken) | ✅ done |
| Cosine LR + warmup + grad-clip | ✅ done |
| Checkpoint save/resume | ✅ done |
| C# WPF training monitor (live train/val loss) | ✅ done |
| 80M-param-class pre-training run on FineWeb-Edu | 🚧 in progress |
| WMMA-tiled FA2 (real speedup, currently per-row is slower than naive at T≤1024) | ⏳ planned |
| BF16 activation buffers in `ModernGPT` (kernels exist; integration deferred until WMMA-FA2) | ⏳ planned |
| Sampling / inference UX | ⏳ basic exists, polishing pending |

All 15 unit tests pass. The model has been validated to converge: a 50M-param run on FineWeb-Edu reached loss **4.74** (train) / **4.78** (val) in 2300 steps.

---

## Architecture

Pre-norm decoder-only transformer (LLaMA-style):

```
tok_emb → N × [ RMSNorm → MHA(GQA + RoPE + QK-Norm) → +residual
              → RMSNorm → SwiGLU FFN → +residual ]
        → RMSNorm → tied LM head (tok_emb^T)
```

| Component | File |
|---|---|
| Tensor (Host/CUDA, FP32/BF16/INT32, views, alloc) | [`llm/src/core/tensor.h`](llm/src/core/tensor.h) |
| cuBLAS row-major GEMM (FP32 + BF16 via `cublasGemmEx`) | [`llm/src/core/gemm.cu`](llm/src/core/gemm.cu) |
| Pre-allocated scratch arena (no `cudaMalloc` in hot path) | [`llm/src/core/scratch.cu`](llm/src/core/scratch.cu) |
| Linear (FP32, BF16-arena, full BF16-in/out) | [`llm/src/model/linear.cu`](llm/src/model/linear.cu) |
| Multi-head attention (split/merge + repeat-KV for GQA) | [`llm/src/model/attention.cu`](llm/src/model/attention.cu) |
| RoPE (precomputed cos/sin tables, in-place fwd+bwd) | [`llm/src/model/rope.cu`](llm/src/model/rope.cu) |
| RMSNorm (no β, in-place-safe backward) | [`llm/src/model/rmsnorm.cu`](llm/src/model/rmsnorm.cu) |
| SwiGLU FFN (fused `silu * up`) | [`llm/src/model/activations.cu`](llm/src/model/activations.cu) |
| Flash-Attention fwd/bwd (per-row, O(N) memory, FP32) | [`llm/src/model/flash_attn.cu`](llm/src/model/flash_attn.cu) |
| Embedding (gather fwd, scatter-add bwd, BF16 variants) | [`llm/src/model/embedding.cu`](llm/src/model/embedding.cu) |
| Full model + checkpointing recompute | [`llm/src/model/modern_gpt.cu`](llm/src/model/modern_gpt.cu) |
| AdamW (fused, FP32 + BF16-states) | [`llm/src/train/adamw.cu`](llm/src/train/adamw.cu) |
| Loss (softmax+CE with `dlogits` in one pass) | [`llm/src/train/loss.cu`](llm/src/train/loss.cu) |
| Grad utils (norm + clip) | [`llm/src/train/grad_utils.cu`](llm/src/train/grad_utils.cu) |
| Checkpoint save/load | [`llm/src/train/checkpoint.cu`](llm/src/train/checkpoint.cu) |
| Training app | [`llm/apps/train_modern_gpt.cu`](llm/apps/train_modern_gpt.cu) |
| Inference app | [`llm/apps/infer_modern_gpt.cu`](llm/apps/infer_modern_gpt.cu) |

---

## Build & test

Requires CUDA Toolkit 12.x, CMake 3.24+, MSVC 2022+. Targets Ampere (sm_80, sm_86) by default.

```cmd
scripts\build_and_test.bat           :: incremental build + ctest
scripts\build_and_test.bat clean     :: wipe build dir first
```

Test suite (15 tests, ~4 s):

```
tensor        Host/CUDA alloc, fill, view, BF16 round-trip
gemm          cuBLAS forward correctness incl. all trans_a/trans_b combos
adamw         quadratic convergence + weight-decay pull-to-zero (FP32 + BF16 states)
loss          softmax+CE forward vs host + finite-diff dlogits
embedding     gather fwd + scatter-add bwd + finite-diff dweight (FP32 + BF16)
linear        forward + finite-diff dx, dW, db (FP32 + BF16 variants)
layernorm     forward vs host + finite-diff dx, dgamma, dbeta
attention     causal scaled-dot forward + finite-diff dq, dk, dv
activations   GELU + SiLU·mul forward exact + finite-diff dx (FP32 + BF16)
mha           split/merge inverses + multi-head matches host reference
rope          in-place rotation matches host + finite-diff backward (FP32 + BF16)
rmsnorm       forward vs host + finite-diff dx, dgamma (FP32 + BF16)
bf16          cast round-trip + linear backward parity vs FP32 reference
flash_attn    forward + logsumexp consistency + backward dQ/dK/dV vs naive
bf16_kernels  add_inplace / silu_mul / rmsnorm / rope BF16 cross-checks
```

Every learnable kernel has analytic backward verified against finite-difference.

---

## Quick start

**1. Prepare a corpus.** Tinyshakespeare for smoke runs:

```cmd
python scripts\download_shakespeare.py
python scripts\prepare_tokens.py --input-text data\tinyshakespeare.txt --output data\tinyshakespeare_cl100k.bin
```

Or a real corpus from HuggingFace (streams, no local download):

```cmd
python scripts\prepare_tokens.py ^
    --hf-dataset HuggingFaceFW/fineweb-edu --hf-config sample-10BT ^
    --max-tokens 1800000000 ^
    --output data\finewebedu_18B_cl100k.bin
```

**2. Train.** A 50M-param Tier-3 setup that converges on FineWeb-Edu:

```cmd
llm\build\train_modern_gpt.exe ^
    --data data\finewebedu_18B_cl100k.bin ^
    --batch 32 --seq-len 512 ^
    --d-model 384 --n-heads 6 --n-kv-heads 2 --n-layers 6 --d-ffn 1536 ^
    --steps 10000 --warmup 200 --log-every 50 ^
    --bf16 --opt-bf16 ^
    --val-every 500 --save-every 500 --val-frac 0.05 ^
    --save-path runs\finewebedu_50m.ckpt ^
    --log runs\finewebedu_50m.jsonl
```

Resume an interrupted run by adding `--resume runs\finewebedu_50m.ckpt`.

**3. Sample.** Once a checkpoint exists:

```cmd
python scripts\sample.py ^
    --ckpt runs\finewebedu_50m.ckpt ^
    --meta data\finewebedu_18B_cl100k.bin.meta ^
    --d-model 384 --n-heads 6 --n-kv-heads 2 --n-layers 6 --d-ffn 1536 --seq-len 512 ^
    --prompt "Photosynthesis is" --num 200 --temp 0.7
```

**4. Monitor live (optional).** The C# WPF monitor under [`gui/Monitor/`](gui/Monitor/) tails the JSONL log file and renders live train/val loss curves:

```cmd
scripts\build_gui.bat
gui\Monitor\bin\Release\net8.0-windows\ModernLLM.Monitor.exe
```

---

## Repository layout

```
modern-llm-v1/
  llm/                ← C++/CUDA training engine
    src/
      core/           ← tensor, gemm, scratch arena, random, cast, dtype, cuda_check
      model/          ← embedding, linear, attention, flash_attn, rope, rmsnorm,
                        activations, modern_gpt (full model)
      train/          ← adamw, loss, grad_utils, checkpoint
      tokenizer/      ← char tokenizer (BPE handled in Python prepare_tokens)
      data/           ← binary token loader + .meta parser
    apps/             ← train_bigram, train_minigpt, train_modern_gpt, infer_modern_gpt
    tests/            ← per-kernel correctness + finite-diff grad checks + benches
    CMakeLists.txt
  gui/Monitor/        ← C# .NET 8 WPF training monitor (LiveCharts2 + MVVM)
  scripts/            ← build_and_test.bat, prepare_tokens.py, sample.py, build_gui.bat
  configs/            ← model configs (JSON)
  data/               ← downloaded corpora (.bin gitignored, .txt kept)
  runs/               ← training JSONL logs + checkpoints (gitignored)
  todo/               ← detailed plan docs (tier3_plan.md, done/, open/, 09_sources.md)
```

---

## Detailed plan + status

See [`todo/tier3_plan.md`](todo/tier3_plan.md) for the full Tier-3 SOTA-Small implementation plan with per-stage results, smoke-run loss tables, and engineering trade-offs. See [`TODO.md`](TODO.md) for what's still open.

Reference list (papers + blogs): [`todo/09_sources.md`](todo/09_sources.md).

---

## License

MIT — see [`LICENSE`](LICENSE).
