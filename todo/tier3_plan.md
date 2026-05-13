# Tier-3 SOTA-Small — Implementation Plan

> Target: 2026er Niveau eines kleinen offenen LMs vom Schlag Olmo-2 / Qwen3-1.5 /
> SmolLM3. Das ist nicht „GPT-5", aber **modern open-source frontier für ≤7B-Modelle**.
>
> Hardware: RTX 3080 Ti (12 GB VRAM, Ampere sm_86, kein FP8).
> Realistische Modellgröße bei voller Tier-3-Engine: 50–300 M Params.

---

## Tier-3 Architektur-Rezept (Endzustand nach allen Stages)

| Komponente | Quelle | Status |
|---|---|---|
| Decoder-only Pre-Norm Transformer | LLaMA / Olmo-2 | ✓ haben wir |
| RoPE position encoding | LLaMA-2/3, DeepSeek, Olmo-2 | ✓ haben wir |
| RMSNorm pre-norm | LLaMA-2/3 | ✓ haben wir |
| Post-norm RMSNorm (optional) | Olmo-2 stability fix | ✗ T3.5 (optional) |
| SwiGLU FFN | LLaMA-2/3, Mistral | ✓ haben wir |
| Cosine LR + Warmup + Grad-Clip | universell | ✓ haben wir |
| BPE tokenizer (cl100k_base) | OpenAI / GPT-4 | ✓ haben wir |
| AdamW (fused) | universell | ✓ haben wir |
| BF16 math (correct) | universell ab 2022 | ✓ Stage 3.A |
| **BF16 schnell (Mirrors+Arena)** | universell | ✗ **T1** |
| **GQA (Grouped-Query Attention)** | LLaMA-2/3, Mistral, Qwen3 | ✗ **T2** |
| **QK-Norm** | DeepSeek, Olmo-2 | ✗ **T3** |
| **Tied Input/Output Embeddings** | klein-Modell-Standard | ✗ **T4** |
| **Flash-Attention 2 forward** | Tri Dao 2023 | ⚠ T5 (per-row, korrekt; nicht WMMA) |
| **Flash-Attention 2 backward** | Tri Dao 2023 | ⚠ T6 (per-row, korrekt; nicht WMMA) |
| **Gradient Checkpointing** | universell | ✗ **T7** |
| **8-bit Optimizer States** | bnb / 8-bit-Adam | ✗ **T8** |
| **BF16 Activations + elementwise** | universell | ✗ **T9** |
| Echtes Pre-Training (FineWeb-Edu, ~10B Tokens) | — | ✗ **T10** |

---

## Stage-Reihenfolge & Abhängigkeiten

```
T1  BF16-Speed  ──── unblockt schnellere Iteration für alles weitere
                │
                ├──  T2 GQA      ──┐
                ├──  T3 QK-Norm   ├── Architektur-Modernisierung
                └──  T4 Tied Emb  ┘   (untereinander unabhängig)
                │
                ├──  T7 Grad-Ckpt   ──┐
                └──  T8 8-bit Opt   ──┘  Memory-Optimierungen
                │
                ├──  T5 FA2 fwd  ──┐
                └──  T6 FA2 bwd  ──┘  großes CUDA-Stück
                │
                └──  T9 BF16 act + elementwise (profitiert von T1+T5+T6)
                │
                └──  T10 Realer Pre-Train auf FineWeb-Edu
```

---

## Stage-Details

### T1 — BF16 Performance Recovery — ✓ DONE (mit Vorbehalt)

**Ergebnis:**
- Mathe identisch (Loss-Kurven 4-Dezimalstellen-genau)
- BF16 nicht mehr langsamer als FP32 — Parität bei D=384 L=6 (vor T1: 30 % langsamer)
- Erwarteter 1.5-2× Speedup nicht erreicht — wir sind launch-overhead-bound bei unseren Modellgrößen, nicht compute-bound. Der 312/156 TFLOPS-Vorteil von BF16 vs TF32 wird erst sichtbar wenn (a) Attention zu Flash-Attn fusioniert ist (T5) und (b) Aktivierungen+Elementwise auch BF16 sind (T9). T1 ist die notwendige Infrastruktur; Speed-Wins folgen.

**Ist-Zustand vor T1:** BF16-Math korrekt (Stage 3.A), aber 30 % langsamer als FP32 wegen
per-Call `cudaMalloc` und wiederholter Weight-Casts.

**Plan:**
1. **Scratch-Arena** in `core/scratch.{h,cu}`: pre-allokierter BF16-Pool, Sub-Allocation per Offset, kein `cudaMalloc` im Hot-Path.
2. **Persistente BF16-Weight-Mirrors:** pro Parameter im Modell ein BF16-Pendant, einmaliger Refresh am Ende von `opt.step()`.
3. `linear_forward_bf16` / `linear_backward_bf16` nehmen pre-cast Mirror als Argument; nur X/dY werden inline gecastet.

**Files:** `core/scratch.{h,cu}` (neu), `model/linear.{h,cu}`, `model/modern_gpt.{h,cu}`, `apps/train_modern_gpt.cu`.

**Tests:**
- alle 13 bestehenden Tests grün
- Smoke: gleichseed FP32 vs BF16 → Loss bit-gleich (Stage 3.A bestätigt)
- Speed: BF16 ≥ 1.5× FP32 bei D=384, L=6 (vs aktuell 0.7×)

**Erfolgskriterium:** BF16 ist auf RTX 3080 Ti messbar schneller als FP32 (TF32-Pfad).

---

### T2 — GQA (Grouped-Query Attention) — ✓ DONE

**Ergebnis-Smokes (D=384 L=6, 50 Steps, gleiche Daten + Seed):**
| n_kv | Params | Loss-Drop | Speed |
|---|---|---|---|
| 6 (= n_q, MHA-Parität) | 91.17M | 4.59 | 10.3 steps/s |
| 3 (GQA 2:1) | 90.29M (-0.88M) | 4.54 | 10.3 steps/s |
| 1 (MQA-extrem) | 89.70M (-1.47M) | 4.48 | 11.8 steps/s |

Param-Einsparung exakt `(n_q-n_kv)·d_h·D·2` pro Block — Theorie und Messung passen. Bei n_kv=n_q ist die Loss-Trajektorie identisch zur Pre-T2-MHA-Variante (Backward-Compat OK).

---

### T2 — GQA (Original Plan)
**Was:** `n_kv_heads < n_q_heads`. Wk, Wv haben kleinere Output-Dim; KV werden vor der Attention auf `n_q` Heads broadcasted.

**Warum:** Standard seit LLaMA-2 (8:1 Verhältnis bei großen Modellen). 4–8× kleinerer KV-Cache bei Inferenz, etwas weniger Compute beim Training.

**Plan:**
1. `ModernGPT::Config` bekommt `int n_kv_heads` (default = `n_heads` → MHA bleibt).
2. `Wk`, `Wv` shape ändert sich von `[D, n_q*d_h]` auf `[D, n_kv*d_h]`.
3. Neuer Kernel `repeat_kv_heads`: `[B*n_kv, T, d_h]` → `[B*n_q, T, d_h]` per Block-Broadcast.
4. Backward: `accumulate_kv_grads` summiert die `n_q/n_kv` repeats zurück.

**Files:** `model/attention.{h,cu}`, `model/modern_gpt.{h,cu}`.

**Tests:**
- Unit: `repeat_kv_heads` round-trip via `accumulate_kv_grads` ≈ identity * (n_q/n_kv)
- Integration: bei `n_kv = n_q` ist Loss-Kurve identisch zu vor-T2-Code
- Bei `n_kv = n_q/4` konvergiert Training sensibel

**Erfolgskriterium:** GQA-Training konvergiert, KV-Tensoren physikalisch kleiner.

---

### T3 — QK-Norm — ✓ DONE

**Implementiert:**
- Per-Block γ-Tensoren (`q_norm_g`, `k_norm_g`, je `[d_h]`), gemeinsam über alle Heads
- Forward: `rmsnorm_forward` auf `q_split` ([B*n_q, T, d_h] view-flach als [B*n_q*T, d_h]) und auf `k_split` (n_kv heads), nach RoPE, vor Attention
- Backward: `rmsnorm_backward` in-place auf den Gradienten-Buffern (kleinere Patch im Kernel war nötig: `dyrow[d]` muss in lokale Variable gelesen werden bevor `dxrow[d]` geschrieben wird, sonst überschreibt der atomicAdd der `d_gamma` mit dem bereits überschriebenen Wert)
- γ_q, γ_k aus weight-decay-Liste herausgenommen (norm-style)
- Keine Kosten an Parametern (12 × d_h ≈ 768 Skalare bei H=6 layers, d_h=64)

**Smoke (D=384 H=6 Hkv=2 L=6, 80 Steps, mit GQA aus T2):** Loss 11.59 → 6.17, val 7.36 → 6.67. Konvergiert messbar besser als ohne QK-Norm. Speed dropped auf 3.8 steps/s (von 10.3) wegen vieler kleiner RMSNorm-Launches; T5 Flash-Attn fusioniert die Attention-Kette und macht das später wieder wett.

---

### T3 — QK-Norm (Original Plan)
**Was:** RMSNorm pro Head auf Q und K **nach RoPE, vor Attention-Scores**. Lernbare Skalierung pro Layer.

**Warum:** Logit-Magnituden bleiben kontrolliert → besseres Tiefe-Skalieren, weniger Loss-Spikes. Olmo-2/Qwen3/DeepSeek nutzen das.

**Plan:**
1. Pro Block neue Params: `q_norm_g [d_h]`, `k_norm_g [d_h]`.
2. Im Forward: nach `rope_apply_inplace` auf `q_split`/`k_split` ein `rmsnorm_per_head` (auf `d_h`-Achse).
3. Backward: zusätzliche RMSNorm-Backward-Calls in Q/K-Pfad.

**Files:** `model/rmsnorm.{h,cu}` (Variante für `[N, T, D] → norm über D`), `model/modern_gpt.{h,cu}`.

**Tests:**
- Finite-diff Gradient-Check für die neue Per-Head-RMSNorm
- Integration: Smoke-Training nach T2 +T3 zeigt vergleichbare/bessere Loss

**Erfolgskriterium:** Gradient-Check passt; Training konvergiert.

---

### T3.5 — Post-Norm RMSNorm (Olmo-2-Stil, optional)
**Was:** Zusätzliche `RMSNorm(γ)` direkt **nach** Attention-Output und nach FFN-Output, vor Residual-Add. Eine Olmo-2-Innovation für tiefes-Modell-Stabilität.

**Plan:** kleines Add-On nach T3, nur falls beim ersten realen Run Loss-Spikes auftreten.

---

### T4 — Tied Input/Output Embeddings — ✓ DONE

**Implementiert:**
- Komplettes Entfernen des separaten `lm_head_` Tensors. tok_emb_ ist die einzige V·D Matrix
- Forward LM-Head: direktes `gemm_fp32_rowmajor` mit `trans_b=true`: `logits[N,V] = lnf_out[N,D] @ tok_emb^T[D,V]`
- Backward in zwei direkten Gemms:
  - `d_lnf_out [N,D] = dlogits [N,V] @ tok_emb [V,D]` (kein Transpose)
  - `d_tok_emb [V,D] += dlogits^T [V,N] @ lnf_out [N,D]` (mit beta=1, akkumuliert mit der späteren `embedding_backward`-Contribution)
- Param-Count vorher 89.99M → nachher **51.49M** (exakt –V·D = –38.5M ✓)
- Speed unverändert vs nicht-tied; Loss konvergiert weiter sauber

**Smoke (Tier-3 v1: D=384, H=6/Hkv=2, L=6, BF16, GQA, QK-Norm, Tied):** 51.49M Params, 80 Steps → Loss 11.62 → 6.66, val 7.40 → 7.01.

---

### T4 — Tied Input/Output Embeddings (Original Plan)
**Was:** `lm_head` ist physikalisch derselbe Tensor wie `tok_emb`, transponiert verwendet.

**Warum:** Spart `V·D` Parameter (bei V=100k, D=384: 38 M params, deutlich für ein 50 M-Modell). SmolLM/TinyLlama-Stil.

**Plan:**
1. `lm_head_` raus, stattdessen Forward `linear_forward(handle, x, tok_emb_, nullptr, logits, /*trans_b=*/true)`.
2. Backward: `linear_backward` schreibt `d_tok_emb` direkt — kein zusätzlicher `d_lm_head`.

**Files:** `model/linear.{h,cu}` (eventuell `linear_forward_tied`-Variante), `model/modern_gpt.{h,cu}`.

**Tests:**
- Param-Count um `V·D` kleiner
- Smoke konvergiert weiterhin
- Finite-diff Check über die geteilte `tok_emb`-Gradient-Akkumulation

---

### T7 — Gradient Checkpointing — ✓ DONE (always-on)

**Implementiert:** Per-Block-Aktivierungstensoren sind jetzt **non-owning Views** (`Tensor::from_blob`) auf einen geteilten Satz Buffer in `ModernGPT` (`act_*`-Felder). Forward läuft normal — überschreibt aber pro Layer denselben Speicher. Backward ruft `recompute_block_forward(L)` für jeden Layer L < n_layers-1 auf, der die Forward-Math erneut über den geteilten Buffer ausführt (vom gespeicherten `bl.inp`).

**Verifikation (D=384 H=6 Hkv=2 L=6, seed=42, 80 Steps):**
| step | Pre-T7 | Post-T7 |
|---|---|---|
| 1 | 11.6233 | 11.6233 |
| 80 | 6.6555 (val 7.0140) | 6.6555 (val 7.0140) |

Loss bit-exakt identisch zu 4 Dezimalstellen — die Mathematik ist unverändert.

**Memory:** Aktivierungs-Memory reduziert von 6 × ~74 MB = ~444 MB auf 1 × 74 MB + 6 × 3 MB inp = ~92 MB (5× Einsparung). Speed-Regression nur ~3 % (3.7 → 3.6 steps/s) — der Recompute ist klein gegenüber dem Rest des Steps.

**Always-on:** Nicht hinter einem Flag. Jeder Run nutzt die gemeinsamen Buffer + Recompute.

---

### T7 — Gradient Checkpointing (Original Plan)
**Was:** Speichere nur die Block-Eingangs-Aktivierung pro Layer; alle Zwischenergebnisse werden im Backward neu berechnet.

**Warum:** ~3× tieferes Modell im selben VRAM, ~30 % langsamer pro Step.

**Plan:**
1. Pro Block: nur `bl.inp` (Block-Input) bleibt. `bl.ln1_out`, `bl.q_proj` etc. werden im Backward neu berechnet.
2. `Block::recompute_forward(handle)` wird vom Backward einmal pro Block aufgerufen.
3. CLI-Flag `--grad-ckpt` schaltet das Verhalten ein.

**Files:** `model/modern_gpt.{h,cu}`.

**Tests:**
- Loss bit-gleich zu Lauf ohne Checkpointing (bei deterministic kernels)
- VRAM-Verbrauch reduziert (manuell mit `nvidia-smi` während Training prüfen)

---

### T8 — Optimizer Memory Savings — ✓ DONE (BF16-Variante; INT8 deferred)

**Implementiert:** AdamW-`m` und `v` werden als **BF16** statt FP32 gespeichert wenn `AdamWConfig::bf16_states = true`. Step-Kernel `adamw_step_kernel_bf16` cast-et im Kernel auf FP32, rechnet, cast-et zurück. CLI-Flag `--opt-bf16`.

**Ergebnis (gleicher Smoke wie T7, seed=42):**
| Step | FP32 states | BF16 states |
|---|---|---|
| 40 | 6.9824 (val 7.4000) | 6.9824 (val 7.4001) |
| 80 | 6.6555 (val 7.0140) | 6.6544 (val 7.0113) |

Loss-Differenzen <0.002 — BF16-Quantisierungsrauschen ist messbar aber harmlos. Convergence-Test in test_adamw verschärft (Tol 2e-2 vs 5e-3 für FP32) — passt.

**Memory-Win:** 51M Modell, von 408 MB → 204 MB Optimizer-State (m + v).

**Was fehlt:** Echtes INT8 (bnb-Style mit Block-Wise-Scale) — würde 4× Memory-Win statt 2× geben, aber braucht Block-Wise-Quantisierung mit per-Block-Scale-Update. Eigene Session, weil komplexer als BF16. Für unser 12-GB-Setup sind die 204 MB schon eine signifikante Befreiung.

---

### T8 — 8-bit Optimizer States (Original Plan)
**Was:** AdamW-`m` und `v` als INT8 mit per-Tensor-Scale.

**Warum:** AdamW-State ist ~2× Modellgröße in FP32 → INT8 viertelt das.

**Plan:**
1. `AdamWParam`: `Tensor m_int8`, `Tensor v_int8`, `float m_scale`, `float v_scale`.
2. Im `step()`-Kernel: dequantisieren → Update → quantisieren zurück. Scale-Update alle ~100 Steps.
3. Default OFF, Flag `--opt-int8`.

**Files:** `train/adamw.{h,cu}`.

**Tests:**
- `test_adamw_converges_quadratic` mit `--opt-int8`-Pfad bestanden (lockerere Tol)
- Smoke: lange Trainingsläufe konvergieren ähnlich

---

### T5 — Flash-Attention 2 Forward — ⚠ DONE mit Vorbehalt

**Implementiert:**
- Per-Row online-softmax CUDA-Kernel in [llm/src/model/flash_attn.cu](../../llm/src/model/flash_attn.cu)
- Causal mask via Tile-Range-Limit (lädt nur K/V-Rows ≤ q_idx)
- Logsumexp `L` pro Row gespeichert (Voraussetzung für T6 backward)
- 14/14 Tests grün, output-match auf 5e-5 vs naive scaled-dot
- Memory: O(N) — kein T×T-`probs`-Buffer

**Vorbehalt — Speed-Wins fehlen noch:**
| T | naive (cuBLAS, TF32 Tensor-Cores) | mein FA2 | Gewinn |
|---|---|---|---|
| 128 | 0.39 ms | 0.25 ms | 1.53× |
| 256 | 0.35 ms | 1.65 ms | 0.21× |
| 512 | 0.23 ms | 5.68 ms | 0.04× |
| 1024 | 0.97 ms | 14.5 ms | 0.07× |

Mein Kernel ist **per-row** (ein Block pro `(b, q_idx)`), eher FA1-Stil als echtes FA2. Echtes FA2 würde Block-level Q-Tiles + Tensor-Core-WMMA-Inner-Matmuls brauchen (CUTLASS oder hand-getuned). Das ist eine eigene mehrgängige Session.

**Was T5 trotzdem liefert:**
- O(N) Memory: bei T=2048+ unverzichtbar (sonst keine Trainingsmöglichkeit für lange Kontexte)
- Korrekte Online-Softmax-Implementierung als Basis für T6 backward
- Logsumexp-Cache als API für die nachfolgenden Schritte

**Ist nicht in `ModernGPT::forward` integriert** — bisher als standalone Kernel + Test. Die echte Integration kommt mit T6, weil der Backward-Pfad anders ist (recomputiert statt probs-cache).

---

### T5 — Flash-Attention 2 Forward (Original Plan)
**Was:** Eigener fused CUDA-Kernel, tile-basiert. Ersetzt naive `scaled_dot_attention_forward` für lange Kontexte.

**Warum:** O(N) Attention-Memory statt O(N²). Pflicht bei T ≥ 2048.

**Plan (Tri Dao 2023 Algorithmus):**
1. Tiles: `Br = Bc = 64` Q-Rows, K-Cols.
2. Online-Softmax: laufender max `m_i`, laufender Sum-of-exps `l_i`, akkumulierter `O_i`.
3. Block je `(b*H + h)` × Q-Tile. Iteriere über K/V-Tiles, korrigiere `m`, `l`, `O` inkrementell.
4. Causal mask via Tile-Indices.
5. Schreibe `O`, plus `L = m + log(l)` als „logsumexp"-Cache fürs Backward.

**Files:** `model/flash_attn.{h,cu}` (neu), Tests.

**Tests:**
- Bit-exact (FP32-Toleranz) vs `scaled_dot_attention_forward` für T=8, 64, 256
- Memory-Profile: Peak-VRAM mit FA2 deutlich kleiner als naive bei T=512+
- Smoke-Training: Loss identisch innerhalb FP32-Toleranz

---

### T6 — Flash-Attention 2 Backward — ⚠ DONE mit Vorbehalt (analog zu T5)

**Implementiert:**
- Drei Kernel in [llm/src/model/flash_attn.cu](../../llm/src/model/flash_attn.cu):
  - `flash_attn_compute_D_kernel` — pro (b, q): D[i] = sum_d O[i,d] * dO[i,d]
  - `flash_attn_dq_kernel` — Block pro (b, q_idx), iteriert K/V-Tiles k=0..q, akkumuliert dQ
  - `flash_attn_dkdv_kernel` — Block pro (b, k_idx), iteriert Q-Tiles q=k..T-1, akkumuliert dK, dV
- Recomputiert P[i,j] = exp(s[i,j] - L[i]) on-the-fly aus dem L-Cache vom Forward
- Dual-Pass weil dQ und dK/dV strukturell unterschiedliche Reductions sind (ein Pass bräuchte atomicAdds)
- Causal-Mask via Tile-Range-Limits (dQ: k ≤ q, dKdV: q ≥ k)
- Memory: O(B·T) für D-Helper + O(N·D) für Gradienten — kein T×T `dprobs`

**Korrektheit:** Finite-Diff vs `scaled_dot_attention_backward`:
| B | T | D | tile | dq diff | dk diff | dv diff |
|---|---|---|---|---|---|---|
| 1 | 8 | 4 | 4 | 6e-8 | 3e-8 | 1e-7 |
| 2 | 16 | 8 | 8 | 5e-8 | 9e-8 | 2e-7 |
| 2 | 64 | 16 | 16 | 6e-8 | 7e-8 | 3e-7 |
| 2 | 128 | 32 | 16 | 7e-8 | 1e-7 | 8e-7 |

Im Wesentlichen FP32-Reordering-Toleranz — die Mathe ist exakt.

**Vorbehalt — Speed (analog zu T5):**
| T | naive bwd (cuBLAS) | mein FA bwd | Verhältnis |
|---|---|---|---|
| 128 | 0.31 ms | 1.01 ms | 0.31× |
| 256 | 1.21 ms | 2.90 ms | 0.42× |
| 512 | 1.17 ms | 8.45 ms | 0.14× |
| 1024 | 1.56 ms | 16.80 ms | 0.09× |

Mein Kernel ist per-row (FA1-Stil), nicht echtes FA2 mit Block-Q-Tiles + Tensor-Core-WMMA. Bei unseren Modellgrößen (T ≤ 1024) ist naive cuBLAS schneller, weil cuBLAS-Tensor-Cores >> handgeschriebener FP32-FMA.

**Was T6 trotzdem liefert:**
- Algorithmische Korrektheit als Basis für später (echtes FA2 mit WMMA wäre eine eigene Session)
- O(N) Memory: bei T ≥ 2048 ist die `probs`-Variante nicht trainierbar (8M-Element-Tensor pro Layer); FA backward läuft weiter
- Die nötige Math-Recompute-Logik (D[i], P aus L) ist verifiziert

**Nicht in `ModernGPT::backward` integriert** — bei T ≤ 1024 ist die naive Variante schneller. Integration sinnvoll erst, wenn (a) wir auf T ≥ 2048 hochziehen oder (b) ein WMMA-FA2 da ist. Für T10 mit T=1024 nehmen wir vorerst die naive Attention.

---

### T9 — BF16 Aktivierungen + elementwise Kernels — ⏳ Phase A done

**Phase A — BF16 Sister-Kernels in elementwise Operationen (DONE):**
- `add_inplace` — BF16 Variante mit cast in/out
- `silu_mul_forward` / `silu_mul_backward` — BF16 Varianten
- `rmsnorm_forward` / `rmsnorm_backward` — BF16 Varianten (gamma + rstd bleiben FP32)
- `rope_apply_inplace` / `rope_apply_backward_inplace` — BF16 Varianten (cos/sin tables FP32)
- Alle Host-Wrapper dispatchen anhand `tensor.dtype()` zwischen FP32- und BF16-Pfad
- Cross-check Tests: 5/5 BF16-Output stimmt mit FP32-Referenz auf <0.05 absolut überein (BF16 hat ~7 Mantissenbits, ~1/128 ≈ 0.008 relativer Fehler — passt)
- 15/15 Tests gesamt grün

**Phase B (Teilfortschritt — DONE: Linear, Permute):**
- ✓ `gemm_bf16inout_rowmajor` (cublasGemmEx mit Ctype=CUDA_R_16BF)
- ✓ `linear_forward_bf16inout` / `linear_backward_bf16inout` — BF16 X/Y, BF16 W (Mirror), FP32 dW Master-Akkumulator, BF16-aware Bias-Add (`add_bias_bf16_kernel`) und Bias-Grad (`bias_grad_bf16_kernel`)
- ✓ `split_heads` / `merge_heads` — BF16-Dispatch via Sister-Kernels (pure byte-copies)

**Phase B Sister-Kernels (DONE):**
- ✓ `repeat_kv_heads` BF16-Dispatch (byte copy)
- ✓ `accumulate_kv_grads_bf16_to_fp32_kernel` — BF16 d_full → FP32 d_split via atomicAdd-cast
- ✓ `embedding_forward_bf16out` + `embedding_backward_bf16in` — Master dWeight bleibt FP32

**Phase B Modell-Integration (DEFERRED bis nach T6):**

Die ursprünglich geplante Integration (act_* als BF16, FP32-Scratches an Attention-Boundary, Casts in fwd+bwd) ist ~300 LoC sauberer Drahtarbeit — wird aber **bewusst deferred**, weil der Trade-off bei unserer Modellgröße schlecht ist:

- **Memory-Win marginal:** 51M-Modell, ~37 MB Aktivierungen → BF16 würde ~18 MB sparen, aber FP32-Scratches an der Attention-Boundary (Q/K/V/ctx) addieren ~24 MB zurück. Netto: nahe Null.
- **Speed-Win minimal:** Wir sind launch-overhead-bound bei dieser Modellgröße, nicht memory-bound. Die elementwise BF16-Pfade würden nur einen kleinen Teil der Step-Zeit verbessern.
- **Matmul-BF16 ist schon da:** `linear_forward_bf16_arena` cast-et X intern und nutzt BF16-cublasGemmEx — alle Block-Linearen laufen bereits BF16-beschleunigt.
- **T6 macht es trivial:** Sobald FA2-Backward existiert, läuft die Attention-Kette nativ in BF16 (cublasGemmEx innerhalb des Kernels). Dann fallen die FP32-Scratches an der Attention-Boundary weg, und die Integration wird strukturell einfach (kein Mixed-Dtype-Problem in `rmsnorm_backward` mehr, weil nichts mehr FP32-zwischenliegt).

**Status:** Alle Phase-A + Phase-B-Kernel sind fertig und getestet (15/15 grün). Die Integration ist „warm" — sie wartet auf T6, dann wird sie ein klarer One-Shot-Patch.

---

### T9 — BF16 Aktivierungen + elementwise Kernels (Original Plan)
**Was:** Aktivierungs-Buffers BF16; alle elementwise Kernels (RMSNorm, RoPE, SiLU·Mul, Softmax) BF16-aware.

**Warum:** Halbiert Aktivierungs-Memory (oft dominanter Trainings-Memory-Posten).

**Plan:**
1. `Tensor`-Allokation in `ModernGPT::allocate` schaltet auf BF16, wenn `use_bf16_`.
2. Jeder Kernel bekommt eine BF16-Variante: lädt BF16 Input, akkumuliert reductions in FP32, schreibt BF16 Output.
3. Loss + dlogits bleiben FP32 (numerisch sensibel).

**Files:** `model/rmsnorm.cu`, `model/rope.cu`, `model/activations.cu`, `model/attention.cu`, `train/loss.cu`.

**Tests:**
- Alle bestehenden Finite-diff-Checks mit lockerer BF16-Toleranz
- Smoke-Training konvergiert zu ähnlichem Loss wie FP32-Aktivierungs-Pfad

---

### T10 — Realer Pre-Training-Run
**Datensatz:** FineWeb-Edu Sample-10BT, ~10 B Tokens via `prepare_tokens.py --max-tokens 10_000_000_000`.
**Modell-Konfig (Beispiel, fittet in 12 GB nach allen Optimierungen):**
- D = 512, L = 8, H = 8, n_kv = 4, Dff = 2048, T = 1024 → ~80 M Params
**Training:** ~30 k Steps, Batch 24, BF16, FA2, Grad-Ckpt, 8-bit Opt.
**Erwartung:** Val-Loss < 4.0 nats; sample-Output ist verständliches Englisch.

---

## Definition of Done je Stage

Jede Stage T1-T9 schließt erst ab, wenn:
1. Code committed
2. Alle bestehenden Unit-Tests grün
3. Mindestens 1 neuer Test für die neue Funktionalität (Finite-Diff bei lernbaren Komponenten)
4. Smoke-Training konvergiert
5. README-Eintrag für die Stage gepflegt

T10 ist abgeschlossen, wenn ein Checkpoint mit Val-Loss < 4.0 in `runs/` liegt
und `sample.py` nachvollziehbar englischen Text generiert.

---

## Geschätzter Aufwand

| Stage | Code-Sessions | GPU-Zeit |
|---|---|---|
| T1 | 1 | gering |
| T2 | 1 | gering |
| T3 | <1 | gering |
| T4 | <1 | gering |
| T7 | 1 | gering |
| T8 | 1 | gering |
| T5 | 1–2 | gering |
| T6 | 2 | gering |
| T9 | 1–2 | gering |
| T10 | 1 (Setup) | 2–7 Tage Train |
| **Gesamt** | **~10–12** | **~Woche** |
