# TODO

Open work, ordered roughly by impact. See [`todo/tier3_plan.md`](todo/tier3_plan.md) for the full Tier-3 plan with rationale and prior results.

---

## Now / blocking

### Finish the FineWeb-Edu pre-training run

Current state: 50M-param Tier-3 setup (D=384, L=6, H=6, n_kv=2, T=512) converges cleanly on FineWeb-Edu. Last training session reached step 2300 / 10000 with loss 4.74 (val 4.78) in ~3 h. Run is paused at the latest checkpoint.

- Resume with `--resume runs\finewebedu_50m.ckpt` appended to the same training command (see README quick-start). ~10 h remaining on RTX 3080 Ti.
- After completion: sample with `scripts\sample.py` to qualitatively check output coherence.
- Expected final loss: ~4.0 nats; sample output should be recognizable English.

---

## Engineering improvements (in priority order)

### 1. Real Flash-Attention 2 (WMMA-tiled)

Current FA fwd/bwd is correct but per-row (FA1-style). At T ≤ 1024 the cuBLAS-naive attention is faster because cuBLAS uses Tensor Cores via WMMA and our hand-written kernel doesn't. To unlock the actual speed win:

- Block-tiled Q-tiles (Br rows per block, e.g. 64)
- WMMA matmul intrinsics or CUTLASS for the inner `Q @ K^T` and `P @ V` matmuls
- Re-check correctness vs the existing per-row kernel as the golden reference

Once this lands, the kernel becomes faster than naive at T ≥ 512 and unlocks T=2048+ training (current naive blows up memory with the T×T probs buffer).

### 2. Wire FA into `ModernGPT::forward` / `::backward`

Currently FA is a standalone kernel with its own tests. The model still uses the naive `scaled_dot_attention`. After (1) lands, route per-head attention through FA. Add a flag like `--flash-attn` to gate it.

### 3. BF16 activation buffer integration

All BF16 building blocks exist (`linear_*_bf16inout`, `embedding_*_bf16`, BF16 elementwise sister-kernels, `accumulate_kv_grads_bf16_to_fp32`). What's missing is the wiring in `ModernGPT::allocate` / `forward` / `backward`:

- Allocate `act_*` buffers as BF16 when `use_bf16_`
- Cast boundaries at attention (FP32 inside, BF16 outside) and at the LM head (FP32 for the big V=100k GEMM)
- Mirror in backward

Deferred until (1) lands — once attention is native BF16 via WMMA-FA2, the boundary casts disappear and the integration becomes a one-shot patch instead of ~300 LoC of mixed-dtype plumbing.

### 4. INT8 optimizer states

We have BF16 AdamW states already (2× memory win, `--opt-bf16`). bnb-style INT8 with per-block scale would give 4× memory savings. Useful when scaling to larger models that pressure VRAM. Algorithm: dequantize → step → quantize back, with block-wise scale updates every ~100 steps.

### 5. KV cache for inference

`infer_modern_gpt` currently runs the full T-context forward per generated token. For interactive sampling, cache keys/values per layer and only forward the new token. Drops generation cost from O(T²) per token to O(T).

### 6. Smaller vocab option

V=100k via cl100k_base means the LM head GEMM is ~50 % of step time. A 32k-BPE retokenizer would cut step time noticeably for the same dataset, at the cost of slightly less efficient text representation. Useful for faster iteration during ablation.

### 7. C# Monitor: GPU stats + checkpoint browser

GUI currently shows train/val loss only. Useful additions:
- `nvidia-smi`-based VRAM / GPU util chart
- Checkpoint list with `infer` button (launch sampling for any saved checkpoint)
- Sample-output viewer panel

---

## Bigger / longer-term

### Larger model + longer pre-training

The current Tier-3 plan target was D=512, L=8 (~80M params) on 1.6B tokens (Chinchilla-optimal). Bench showed ~36 s/step at that config → 27-day training; deferred in favor of the smaller D=384, L=6 run that fits an overnight session. After (1) + WMMA-FA2 lands, revisit larger configs.

### Post-Norm RMSNorm (Olmo-2 stability fix)

Optional stability addon: add an extra `RMSNorm(γ)` right after attention output and after FFN output, before residual add. Only worth doing if we see loss spikes during a deep-model run.

### Multi-token prediction (MTP)

DeepSeek-style auxiliary heads that predict tokens further ahead. Improves sample efficiency. Eigene Session.

### MoE (Mixture-of-Experts)

Sparse FFN routing. Big change to FFN code path. Eigene Session.

### MLA (Multi-Head Latent Attention)

DeepSeek-V2-style compressed KV cache via low-rank latent. Would replace GQA. Big architectural shift; defer until single-GPU training of dense GQA is solid first.

### RL post-training pipeline (SFT → DPO → GRPO)

Out of scope of the pre-training engine. Sketched in [`todo/open/04_rl_training.md`](todo/open/04_rl_training.md).

### Publication path

Export to GGUF / safetensors for HuggingFace upload, GitHub release with sample-output gallery. Sketched in [`todo/open/07_publish.md`](todo/open/07_publish.md).

---

## Known limitations

- **Single-GPU only.** No multi-GPU / FSDP / pipeline parallelism. The codebase is single-GPU-now / multi-GPU-aware (no global statics that would block adding it), but distribution is not implemented.
- **Windows-first.** Build scripts are `.bat`. CMakeLists supports Linux but is not regularly tested there.
- **No CPU fallback.** All compute paths are CUDA. Host tensors exist for I/O only.
- **`infer_modern_gpt` has no KV cache.** Slow for long generations.
- **The C# GUI is Windows-only (WPF).** A MAUI port would unlock Mac/Linux.

---

## Contributing

This is a personal learn-everything-from-scratch project — PRs welcome but please open an issue first to discuss direction.
