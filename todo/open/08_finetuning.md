# 08 — Fine-Tuning Plan (Post-Release)

> How to fine-tune ModernLLM after release — in C++ with a LoRA-equivalent, and via HuggingFace for the community.

---

## Fine-Tuning Strategies

```
Strategy 1: Full fine-tune (all weights)      — high quality, expensive
Strategy 2: LoRA-equivalent in C++           — efficient, our primary approach
Strategy 3: HuggingFace-compatible fine-tune  — community can fine-tune via Python
Strategy 4: Domain adaptation (continued PT)  — domain-specific pre-training
```

---

## Strategy 1 — Full Fine-Tuning

Use when: you have significant compute and want best possible quality on a specific domain.

Same as training loop but:
- Start from the released checkpoint
- Use a much lower LR: 1e-6 to 5e-6 (10–50× lower than pre-training)
- Train on domain-specific SFT data
- Use gradient checkpointing to manage memory
- Train for 1–3 epochs only

---

## Strategy 2 — LoRA-Equivalent in C++ (Primary)

LoRA (Low-Rank Adaptation) adds small trainable matrices to frozen base weights. Instead of updating W (large), you update A and B (small, rank-r), where the update is `∆W = A @ B`.

```cpp
// For a weight matrix W of shape [out, in]
// LoRA adds:
//   W_lora_A: [r, in]      << small trainable
//   W_lora_B: [out, r]     << small trainable
// where r << min(out, in), typically r=8 or r=16

struct LoRAAdapter {
    float* A;       // [r, in_features]
    float* B;       // [out_features, r]
    float  scale;   // alpha / r, controls contribution magnitude
    int    r;       // rank
    int    in_features, out_features;
};

// Modified forward for a LoRA-adapted linear layer:
void lora_linear_forward(float* out, const float* x,
                         const float* W_frozen,          // base weights (no grad)
                         const LoRAAdapter& lora,
                         int seq_len) {
    // Base output (no gradient)
    gemm(out, x, W_frozen, seq_len, lora.out_features, lora.in_features);
    
    // LoRA delta (with gradient)
    float* tmp = alloc_temp(seq_len * lora.r);
    gemm(tmp, x,   lora.A, seq_len, lora.r,            lora.in_features);
    gemm(out, tmp, lora.B, seq_len, lora.out_features,  lora.r,
         /*add_to_out=*/true, /*scale=*/lora.scale);
    
    free_temp(tmp);
}
```

### Which Layers to Apply LoRA To

Apply LoRA to attention projections (Q, K, V, O) and optionally FFN:

```cpp
struct LoRAConfig {
    int   rank         = 16;
    float alpha        = 32.0f;     // scale = alpha/rank = 2.0
    float dropout      = 0.05f;
    bool  apply_q      = true;
    bool  apply_k      = true;
    bool  apply_v      = true;
    bool  apply_o      = true;
    bool  apply_ffn    = false;     // optional, adds more params
};
```

### Memory Advantage

For a 300M model with rank=16:
- Full fine-tune: 1.2 GB trainable parameters
- LoRA (Q,K,V,O only): ~10 MB trainable parameters
- Optimizer states: ~20 MB (vs 2.4 GB for full)
- Can fine-tune on a 4GB GPU

### Save Only LoRA Weights

```cpp
void save_lora_checkpoint(const std::string& path,
                          const std::vector<LoRAAdapter>& adapters) {
    // Save only A and B matrices — base model stays unchanged
    // File is tiny: typically 20–80 MB depending on rank and layers
}
```

Published as a separate small file — users download base model + LoRA adapter.

---

## Strategy 3 — HuggingFace-Compatible Fine-Tuning (Community)

Since you publish safetensors + `modeling_modernllm.py`, the community can fine-tune using standard tools:

### With HuggingFace TRL + LoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer, SFTConfig

model = AutoModelForCausalLM.from_pretrained(
    "YOUR_USERNAME/modernllm-300m",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# Apply LoRA via PEFT
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Fine-tune with SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=your_dataset,
    args=SFTConfig(
        output_dir="finetuned_model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
    )
)
trainer.train()
```

**Your job:** Make sure `modeling_modernllm.py` correctly maps parameter names so PEFT can find `q_proj`, `k_proj`, etc.

### With Unsloth (fast fine-tuning)

Unsloth provides 2–4× faster fine-tuning on consumer GPUs. Your model can support it if your `modeling_modernllm.py` follows LLaMA conventions:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "YOUR_USERNAME/modernllm-300m",
    max_seq_length=8192,
    dtype=torch.bfloat16,
    load_in_4bit=True  # QLoRA
)
```

---

## Strategy 4 — Domain Adaptation (Continued Pre-Training)

Fine-tune on domain-specific raw text (not instruction format) at a low LR. Good for specialized domains.

**Examples:**
- Medical: PubMed full-text + medical textbooks
- Legal: court opinions, legislation (FreeLaw from Pile)
- Code: language-specific repositories (Rust, Go, Java)
- German: CC-100 German + German Wikipedia

```cpp
// Same as pre-training loop, but:
// 1. Start from released checkpoint
// 2. LR = 1e-5 (10× lower)
// 3. Dataset = domain-specific raw text
// 4. Train for 1-5B tokens (short run)
// 5. Then SFT on domain instructions
```

---

## Fine-Tuning Dataset Recommendations (Streamable)

### Instruction Fine-Tuning

| Domain | Dataset | HF ID |
|---|---|---|
| General | SmolTalk | `HuggingFaceTB/smoltalk` |
| Medical | MedAlpaca | `medalpaca/medical_meadow_medqa` |
| Code | CodeAlpaca | `sahil2801/CodeAlpaca-20k` |
| Math | MetaMath | `meta-math/MetaMathQA` |
| German | German Alpaca | `bjoernp/alpaca-cleaned-de` |
| Legal | LegalBench | `hazyresearch/legalbench` |

### DPO Fine-Tuning (Preference)

| Dataset | HF ID | Notes |
|---|---|---|
| UltraFeedback | `HuggingFaceH4/ultrafeedback_binarized` | General preference |
| HelpSteer2 | `nvidia/HelpSteer2` | Helpfulness focus |
| Code Feedback | `m-a-p/CodeFeedback-Filtered-Instruction` | Code quality |

---

## Fine-Tuning Schedule (Example: Medical Domain)

```
Week 1: Continued pre-training
  - Data: PubMed abstracts + medical textbooks (raw text)
  - Tokens: 2B
  - LR: 1e-5

Week 2: Medical SFT
  - Data: MedAlpaca + MedQA instruction pairs
  - Samples: 50K
  - LR: 5e-6

Week 3: Medical DPO
  - Data: Medical preference pairs (or generate + score)
  - Pairs: 5K
  - LR: 1e-6

→ Release: modernllm-300m-medical
```

---

## Publishing Fine-Tuned Models

For LoRA adapters:

```
HuggingFace repo: YOUR_USERNAME/modernllm-300m-medical-lora
Files:
  adapter_config.json     ← LoRA config
  adapter_model.safetensors  ← only A/B matrices, ~20MB
  README.md
```

Users load with:
```python
model = AutoModelForCausalLM.from_pretrained("YOUR_USERNAME/modernllm-300m")
model.load_adapter("YOUR_USERNAME/modernllm-300m-medical-lora")
```

For full fine-tunes: publish as a new model repo following the same structure as the base model.
