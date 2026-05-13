# 01 — C++ Model Architecture

> Full build order and design spec for every component of the modern transformer in C++.

---

## Build Order (strict — each phase depends on the previous)

```
Phase 1: Core Infrastructure
Phase 2: Positional Encoding (RoPE)
Phase 3: Attention (MHA → GQA → MLA)
Phase 4: Flash Attention Kernel
Phase 5: FFN (SwiGLU)
Phase 6: MoE Layer
Phase 7: Transformer Block
Phase 8: Full Model (embedding + stack + LM head)
Phase 9: Tokenizer (BPE)
```

---

## Phase 1 — Core Infrastructure

### Tensor Class (`core/tensor.h`)

The foundation everything else builds on.

```cpp
struct Tensor {
    float*   data;         // raw float32 buffer (aligned 64-byte)
    uint16_t* data_bf16;   // bf16 buffer (same memory, reinterpreted)
    
    std::vector<int> shape;
    std::vector<int> strides;   // row-major computed strides
    int              numel;     // total elements
    DType            dtype;     // FP32 | BF16 | FP8
    bool             owns_data; // whether to free on destruction
    
    // Core ops
    Tensor view(std::vector<int> new_shape);
    Tensor slice(int dim, int start, int end);
    void   fill(float val);
    void   zero_();
    float  item();  // scalar extract
};
```

**Key rules:**
- 64-byte aligned allocations (`posix_memalign` or `_aligned_malloc`)
- Strides computed once on construction, no recomputation
- Views share the same data pointer — no copies

### GEMM Kernel (`core/gemm.cpp`)

Build in three stages:

**Stage 1 — Naive:**
```cpp
void gemm_naive(float* C, const float* A, const float* B,
                int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += A[i*K + k] * B[k*N + j];
            C[i*N + j] = acc;
        }
}
```

**Stage 2 — Tiled (cache-friendly):**
- Block into tiles that fit L1/L2 cache
- Typical tile size: 64×64 for L2, 16×16 for L1
- Transpose B before multiplication (B^T is row-major for inner loop)

**Stage 3 — AVX2 SIMD:**
```cpp
// Process 8 floats at once with 256-bit registers
#include <immintrin.h>

void gemm_avx2(float* C, const float* A, const float* B,
               int M, int N, int K) {
    // Inner loop uses _mm256_fmadd_ps (fused multiply-add)
    // 8-wide vectorized accumulation
}
```

For BF16: implement BF16→FP32 upcasting for the inner GEMM, downcast result back.

### Core Ops (`core/ops.cpp`)

```cpp
// RMSNorm (NOT LayerNorm — all modern models use this)
void rms_norm(float* out, const float* x, const float* weight,
              int d_model, float eps = 1e-6f);
// Formula: out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]

// Softmax (numerically stable)
void softmax(float* out, const float* x, int n);
// Step 1: find max, Step 2: exp(x - max), Step 3: normalize

// SiLU activation
inline float silu(float x) { return x / (1.f + expf(-x)); }

// Embedding lookup
void embedding_forward(float* out, const int* token_ids,
                       const float* embed_table,
                       int seq_len, int d_model);
```

---

## Phase 2 — RoPE (`attention/rope.cpp`)

Rotary Position Embeddings — replaces sinusoidal, enables long context.

**Precompute once at model init:**
```cpp
struct RoPECache {
    float* cos_cache;  // [max_seq_len, head_dim/2]
    float* sin_cache;  // [max_seq_len, head_dim/2]
    
    void precompute(int max_seq_len, int head_dim, float base = 10000.f) {
        for (int pos = 0; pos < max_seq_len; pos++) {
            for (int i = 0; i < head_dim/2; i++) {
                float freq = 1.f / powf(base, (2.f*i) / head_dim);
                float angle = pos * freq;
                cos_cache[pos*(head_dim/2) + i] = cosf(angle);
                sin_cache[pos*(head_dim/2) + i] = sinf(angle);
            }
        }
    }
};
```

**Apply at attention time (in-place, rotates Q and K):**
```cpp
void apply_rope(float* q, float* k,
                const RoPECache& cache,
                int pos, int n_heads, int head_dim) {
    // For each head, rotate pairs of dimensions:
    // [x0, x1] → [x0*cos - x1*sin, x0*sin + x1*cos]
    for (int h = 0; h < n_heads; h++) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < head_dim/2; i++) {
            float cos_val = cache.cos_cache[pos*(head_dim/2) + i];
            float sin_val = cache.sin_cache[pos*(head_dim/2) + i];
            float q0 = qh[2*i], q1 = qh[2*i+1];
            qh[2*i]   = q0*cos_val - q1*sin_val;
            qh[2*i+1] = q0*sin_val + q1*cos_val;
        }
    }
    // Same for k
}
```

**Decoupled RoPE for MLA:** Apply RoPE only to a subset of dimensions (the "rope" portion), not the full head. The non-RoPE portion goes through latent compression.

---

## Phase 3 — Attention Evolution

### MHA (baseline, build first)

```cpp
struct MHAWeights {
    float* W_q;   // [d_model, n_heads * head_dim]
    float* W_k;   // [d_model, n_heads * head_dim]
    float* W_v;   // [d_model, n_heads * head_dim]
    float* W_o;   // [n_heads * head_dim, d_model]
};

// Forward:
// 1. Q = x @ W_q, K = x @ W_k, V = x @ W_v
// 2. Apply RoPE to Q and K
// 3. scores = (Q @ K^T) / sqrt(head_dim)
// 4. Apply causal mask (upper triangle = -inf)
// 5. attn = softmax(scores)
// 6. out = attn @ V
// 7. out = out @ W_o
```

### GQA (intermediate — fewer K/V heads)

```cpp
struct GQAWeights {
    float* W_q;   // [d_model, n_q_heads * head_dim]
    float* W_k;   // [d_model, n_kv_heads * head_dim]   // SMALLER
    float* W_v;   // [d_model, n_kv_heads * head_dim]   // SMALLER
    float* W_o;   // [n_q_heads * head_dim, d_model]
    
    int n_q_heads;
    int n_kv_heads;
    int group_size;  // = n_q_heads / n_kv_heads
};
// In forward: broadcast each KV head to group_size Q heads
```

### MLA (final — DeepSeek-style)

The key innovation: compress KV into a low-dim latent, cache the latent not full KV.

```cpp
struct MLAWeights {
    // Query path
    float* W_dq;    // [d_model, q_lora_rank]     down-project query
    float* W_uq;    // [q_lora_rank, n_heads*qk_nope_dim]  up-project
    float* W_qr;    // [q_lora_rank, n_heads*qk_rope_dim]  RoPE query

    // Key-Value path (SHARED compression)
    float* W_dkv;   // [d_model, kv_lora_rank]    down-project KV (cached!)
    float* W_uk;    // [kv_lora_rank, n_heads*qk_nope_dim]  up-project keys
    float* W_uv;    // [kv_lora_rank, n_heads*v_head_dim]   up-project values
    float* W_kr;    // [d_model, n_heads*qk_rope_dim]       decoupled RoPE key

    // Output
    float* W_o;     // [n_heads*v_head_dim, d_model]

    // Dims
    int d_model, n_heads;
    int q_lora_rank, kv_lora_rank;
    int qk_nope_dim, qk_rope_dim, v_head_dim;
};
```

**KV Cache stores only `c_kv` (compressed latent):**
```cpp
struct MLAKVCache {
    float* c_kv;   // [n_layers, max_seq_len, kv_lora_rank]  << MUCH smaller
    float* k_rope; // [n_layers, max_seq_len, n_heads, qk_rope_dim]
    int    cur_len;
};
```

**Forward pass:**
```
1. c_q = x @ W_dq                        // compress query
2. q_nope = c_q @ W_uq                   // query (non-RoPE part)
3. q_rope = c_q @ W_qr                   // query (RoPE part)
4. apply_rope(q_rope, pos)

5. c_kv = x @ W_dkv                      // compress KV → STORE THIS in cache
6. k_rope = x @ W_kr                     // decoupled key RoPE → STORE in cache
7. apply_rope(k_rope, pos)

8. At attention time, up-project from cache:
   k_nope = c_kv @ W_uk                  // expand keys
   v      = c_kv @ W_uv                  // expand values

9. k = concat(k_nope, k_rope)
   q = concat(q_nope, q_rope)
10. scores = q @ k^T / sqrt(dim)
11. out = softmax(scores) @ v
12. out = out @ W_o
```

**Absorb trick:** Fuse `W_uk` into `W_uq` and `W_uk` into `W_o` so you never materialize full K/V — cached latent is used directly.

---

## Phase 4 — Flash Attention 2 (`attention/flash_attn.cpp`)

Never materializes the full N×N attention matrix. O(N) memory, exact output.

**Algorithm (per block):**
```
Tile Q into blocks of size Br
Tile K, V into blocks of size Bc

For each Q block:
  Initialize: O_i = 0, l_i = 0, m_i = -inf
  
  For each KV block:
    S_ij = Q_i @ K_j^T / sqrt(d)    // [Br, Bc] — fits in SRAM
    Apply causal mask to S_ij
    m_ij = rowmax(S_ij)
    P_ij = exp(S_ij - m_ij)
    l_ij = rowsum(P_ij)
    
    // Running softmax correction
    m_i_new = max(m_i, m_ij)
    l_i = exp(m_i - m_i_new) * l_i + exp(m_ij - m_i_new) * l_ij
    O_i = diag(exp(m_i - m_i_new)) * O_i + P_ij @ V_j
    m_i = m_i_new
  
  O_i = diag(1/l_i) * O_i   // final normalization
```

Tile sizes: `Br = Bc = 64` for typical GPU SRAM. On CPU, size to fit L2 cache.

---

## Phase 5 — SwiGLU FFN (`ffn/swiglu.cpp`)

```cpp
struct SwiGLUWeights {
    float* W_gate;   // [d_model, d_ffn]
    float* W_up;     // [d_model, d_ffn]
    float* W_down;   // [d_ffn, d_model]
};

void swiglu_forward(float* out, const float* x, const SwiGLUWeights& w,
                    int seq_len, int d_model, int d_ffn) {
    // gate = x @ W_gate
    // up   = x @ W_up
    // hidden = SiLU(gate) * up    ← element-wise gating
    // out  = hidden @ W_down
}
```

`d_ffn` is typically `2/3 * 4 * d_model` (rounded to multiple of 64) to maintain param count parity with standard FFN.

---

## Phase 6 — MoE Layer (`ffn/moe.cpp`)

```cpp
struct MoELayer {
    // Router
    float* W_gate;          // [d_model, n_experts]
    
    // Experts (each is a SwiGLU FFN with smaller d_ffn)
    SwiGLUWeights* experts; // array of n_experts
    
    int n_experts;          // e.g. 64
    int n_active;           // top-K active per token, e.g. 8
    float aux_loss_coeff;   // load balancing loss weight (0.001)
};

void moe_forward(float* out, const float* x,
                 MoELayer& moe, int seq_len,
                 float* aux_loss_out) {
    
    // 1. Router: compute logits for each token
    //    router_logits = x @ W_gate   [seq_len, n_experts]
    
    // 2. Top-K selection
    //    For each token: pick top n_active experts by logit
    //    expert_weights = softmax(top_k_logits)
    
    // 3. Dispatch: group tokens by expert
    //    expert_inputs[e] = gather(x, tokens_assigned_to_e)
    
    // 4. Run each expert on its token batch
    //    expert_outputs[e] = swiglu_forward(expert_inputs[e], experts[e])
    
    // 5. Scatter back and weight
    //    out[token] = sum(expert_weight[e] * expert_outputs[e][token])
    
    // 6. Compute aux load-balancing loss
    //    f_i = fraction of tokens routed to expert i
    //    p_i = mean router probability for expert i
    //    aux_loss = n_experts * sum(f_i * p_i)
}
```

---

## Phase 7 — Transformer Block (`model/transformer_block.cpp`)

```cpp
struct TransformerBlock {
    RMSNormWeights norm1, norm2;
    MLAWeights     attn;
    MoELayer       ffn;       // or SwiGLUWeights for dense version
    bool           is_moe;    // alternate: some layers dense, some MoE
};

void block_forward(float* x, TransformerBlock& block,
                   MLAKVCache& kv_cache, int pos) {
    // Pre-norm → Attention → Residual
    float* normed = rms_norm(x, block.norm1);
    float* attn_out = mla_forward(normed, block.attn, kv_cache, pos);
    x = x + attn_out;   // residual

    // Pre-norm → FFN → Residual
    normed = rms_norm(x, block.norm2);
    float* ffn_out = moe_forward(normed, block.ffn);
    x = x + ffn_out;    // residual
}
```

---

## Phase 8 — Full Model (`model/transformer.cpp`)

```cpp
struct ModernLLM {
    // Config
    int d_model, n_layers, n_heads, vocab_size, max_seq_len;
    int n_experts, n_active_experts;
    int kv_lora_rank, q_lora_rank;
    
    // Weights
    float*             token_embed;   // [vocab_size, d_model]
    TransformerBlock*  blocks;        // [n_layers]
    RMSNormWeights     final_norm;
    float*             lm_head;       // [d_model, vocab_size]
    
    // KV cache (inference)
    MLAKVCache         kv_cache;
    
    float forward(int* tokens, int seq_len);
    void  save_checkpoint(const char* path);
    void  load_checkpoint(const char* path);
};
```

---

## Phase 9 — BPE Tokenizer (`tokenizer/bpe.cpp`)

Compatible with tiktoken/HuggingFace tokenizer files so you can load GPT-4 or LLaMA vocab.

```cpp
struct BPETokenizer {
    std::unordered_map<std::string, int> vocab;     // token → id
    std::unordered_map<int, std::string> id_to_tok; // id → token
    std::vector<std::pair<int,int>>      merges;    // BPE merge rules
    
    std::vector<int> encode(const std::string& text);
    std::string      decode(const std::vector<int>& ids);
    
    // Load from HuggingFace tokenizer.json format
    void load(const std::string& path);
};
```

---

## Model Config (JSON)

Each model size is a config file:

```json
{
  "model_name": "modernllm-300m",
  "d_model": 1024,
  "n_layers": 24,
  "n_heads": 16,
  "n_kv_heads": 8,
  "head_dim": 64,
  "d_ffn": 2816,
  "vocab_size": 32000,
  "max_seq_len": 8192,
  "rope_base": 10000.0,
  "n_experts": 64,
  "n_active_experts": 8,
  "kv_lora_rank": 512,
  "q_lora_rank": 1536,
  "qk_rope_dim": 64,
  "qk_nope_dim": 128,
  "v_head_dim": 128,
  "aux_loss_coeff": 0.001,
  "norm_eps": 1e-6
}
```

Scale up configs: 300M → 1B → 3B by increasing `d_model`, `n_layers`, `n_experts`.
