# 02 — Training Efficiency

> Everything needed to make training fast and memory-efficient in C++.

---

## Priority Order

| Priority | Technique | Gain |
|---|---|---|
| 1 | Flash Attention 2 | 2–4× memory, enables long context |
| 2 | BF16 mixed precision | ~2× throughput |
| 3 | Fused kernels | 20–40% |
| 4 | Gradient checkpointing | fits 3× larger model |
| 5 | Gradient accumulation | large effective batch |
| 6 | Async data pipeline | eliminates CPU wait |
| 7 | 8-bit optimizer states | 4× optimizer memory |
| 8 | MTP training objective | better sample efficiency |
| 9 | Cosine LR schedule | stable convergence |

---

## 1. BF16 Mixed Precision

**Why BF16 not FP16:** BF16 has the same exponent range as FP32 (avoids overflow). FP16 overflows frequently during LLM training.

**Strategy:**
- Store model weights in BF16 (half memory)
- Keep FP32 "master copy" of weights for AdamW update
- Forward + backward in BF16
- Cast to FP32 for optimizer step, copy back to BF16

```cpp
// BF16 is just the top 16 bits of a float32
// Conversion: just bitshift
inline uint16_t f32_to_bf16(float f) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    return *reinterpret_cast<float*>(&bits);
}
```

**Memory layout:**
```cpp
struct WeightBuffer {
    uint16_t* bf16_data;    // used in forward/backward
    float*    fp32_master;  // used in AdamW update
    int       numel;
    
    void sync_bf16_from_master() {
        for (int i = 0; i < numel; i++)
            bf16_data[i] = f32_to_bf16(fp32_master[i]);
    }
};
```

---

## 2. Flash Attention 2 (see 01_architecture.md Phase 4)

Already specified in architecture doc. Key point: must be implemented before long-context training — standard attention runs out of memory at 4K+ tokens.

---

## 3. Fused Kernels

**Fused RMSNorm + Linear:**
- Instead of: normalize → write to memory → read → matmul
- Do: normalize in registers → immediately feed into matmul inner loop
- Saves 2 memory round trips

**Fused SwiGLU:**
```cpp
// Naive (3 separate passes):
gate   = x @ W_gate;   // write to memory
up     = x @ W_up;     // write to memory
hidden = silu(gate) * up;  // write to memory
out    = hidden @ W_down;

// Fused (compute gate and up simultaneously, gate never hits RAM):
// Process column-by-column: for each output column of W_gate and W_up,
// compute both projections and gate in the same pass
```

**Fused AdamW:**
```cpp
void adamw_step_fused(float* param, float* grad,
                      float* m, float* v,      // momentum buffers
                      float lr, float beta1, float beta2,
                      float eps, float weight_decay,
                      int step, int numel) {
    // All in one loop — no separate passes for bias correction,
    // weight decay, or parameter update
    float bc1 = 1.f - powf(beta1, step);
    float bc2 = 1.f - powf(beta2, step);
    for (int i = 0; i < numel; i++) {
        m[i] = beta1 * m[i] + (1-beta1) * grad[i];
        v[i] = beta2 * v[i] + (1-beta2) * grad[i]*grad[i];
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        param[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param[i]);
    }
}
```

---

## 4. Gradient Checkpointing

**Problem:** Storing all activations for backward pass uses O(n_layers × seq_len × d_model) memory.

**Solution:** Only store activations at checkpoint boundaries. Recompute everything in between during backward.

```cpp
enum ForwardMode {
    INFERENCE,           // no grad storage
    TRAIN_FULL,          // store all activations (fast backward, high memory)
    TRAIN_CHECKPOINT     // store only at checkpoints (recompute on backward)
};

struct CheckpointManager {
    // Store input to each checkpoint block
    std::vector<Tensor> checkpoints;   // one per checkpoint interval
    int checkpoint_interval = 4;       // recompute every 4 layers
    
    void save_checkpoint(int layer, const Tensor& x);
    Tensor get_checkpoint(int layer);
};
```

**Memory savings:** With checkpoint every K layers, memory is O(N/K) activations + O(K) for recomputation window. Typical K=4 gives 4× memory reduction at ~33% compute overhead.

---

## 5. Gradient Accumulation

```cpp
struct TrainingConfig {
    int batch_size      = 1;    // micro-batch (fits in memory)
    int accum_steps     = 32;   // effective batch = 32
    int effective_batch = batch_size * accum_steps;
};

// Training loop:
zero_gradients();
for (int micro = 0; micro < config.accum_steps; micro++) {
    Batch b = dataloader.next_micro_batch();
    float loss = model.forward(b);
    loss /= config.accum_steps;    // scale loss so gradients are averaged
    model.backward(loss);          // accumulates into grad buffers
}
optimizer.step();                  // single update after all micro-batches
```

---

## 6. Async Data Pipeline

C++ thread pair: producer fills a ring buffer, training loop consumes.

```cpp
struct DataPipeline {
    std::queue<Batch>        buffer;
    std::mutex               mu;
    std::condition_variable  cv;
    std::thread              producer_thread;
    int                      buffer_size = 8;    // prefetch 8 batches
    
    void start(DataLoader& loader) {
        producer_thread = std::thread([&] {
            while (running) {
                Batch b = loader.fetch_and_tokenize_next();
                std::unique_lock lock(mu);
                cv.wait(lock, [&]{ return buffer.size() < buffer_size; });
                buffer.push(std::move(b));
                cv.notify_one();
            }
        });
    }
    
    Batch get_next() {
        std::unique_lock lock(mu);
        cv.wait(lock, [&]{ return !buffer.empty(); });
        Batch b = std::move(buffer.front());
        buffer.pop();
        cv.notify_one();
        return b;
    }
};
```

---

## 7. 8-bit Optimizer States

AdamW's momentum (m) and variance (v) buffers are FP32 by default — that's 2× model size extra memory.

Quantize them to INT8:
- Maintain a per-tensor scale factor
- Quantize: `q = round(x / scale)`, clamp to [-127, 127]
- Dequantize before update: `x = q * scale`
- Update scale periodically (every 100 steps)

This cuts optimizer memory from 8 bytes/param to 2 bytes/param — 4× reduction.

---

## 8. Multi-Token Prediction (MTP)

**Architecture addition:** D extra linear heads on top of final hidden states.

```cpp
struct MTPHeads {
    float** W_heads;   // D extra LM heads [D][d_model, vocab_size]
    int D = 3;         // predict 3 tokens ahead
    float lambda = 0.3; // weight of MTP loss
};
```

**Loss:**
```
L_main = cross_entropy(lm_head(h_t), token_{t+1})
L_mtp  = sum_{d=1}^{D} cross_entropy(mtp_head_d(h_t), token_{t+d+1})
L_total = L_main + lambda * L_mtp
```

**Why it helps:** Each training step extracts more gradient signal from the same data. DeepSeek reported better perplexity at same token count. At inference, discard MTP heads — zero overhead.

---

## 9. Learning Rate Schedule

```cpp
float get_lr(int step, int warmup_steps, int total_steps,
             float max_lr, float min_lr) {
    if (step < warmup_steps) {
        // Linear warmup
        return max_lr * (float)step / warmup_steps;
    }
    // Cosine decay
    float progress = (float)(step - warmup_steps) / (total_steps - warmup_steps);
    return min_lr + 0.5f * (max_lr - min_lr) * (1.f + cosf(M_PI * progress));
}
```

**Recommended values:**
- `max_lr`: 3e-4 (300M), 1e-4 (1B)
- `min_lr`: max_lr / 10
- `warmup_steps`: 1% of total steps
- `weight_decay`: 0.1
- `grad_clip`: 1.0 (clip gradient norm)

---

## 10. Gradient Clipping

Prevent exploding gradients — critical for training stability.

```cpp
void clip_grad_norm(std::vector<Tensor>& grads, float max_norm) {
    float total_norm = 0.f;
    for (auto& g : grads)
        for (int i = 0; i < g.numel; i++)
            total_norm += g.data[i] * g.data[i];
    total_norm = sqrtf(total_norm);
    
    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        for (auto& g : grads)
            for (int i = 0; i < g.numel; i++)
                g.data[i] *= scale;
    }
}
```

---

## Memory Budget (300M model)

| Component | FP32 | BF16 + FP32 master |
|---|---|---|
| Model weights | 1.2 GB | 0.6 GB (BF16) |
| FP32 master copy | — | 1.2 GB |
| Adam m + v states | 2.4 GB | 0.6 GB (INT8) |
| Activations (no checkpoint) | ~8 GB | ~4 GB |
| Activations (checkpoint K=4) | ~2 GB | ~1 GB |
| **Total** | **~14 GB** | **~7 GB** |

BF16 + INT8 optimizer + gradient checkpointing cuts memory from ~14 GB to ~7 GB for a 300M model.
