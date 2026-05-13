# 06 — Evaluation

> How to measure model quality during and after training.

---

## Evaluation Layers

```
Layer 1: Perplexity (continuous — every N steps during training)
Layer 2: Generation quality (manual spot checks during training)
Layer 3: Benchmark suite (formal evaluation at checkpoints)
Layer 4: RL reward tracking (during post-training stages)
```

---

## Layer 1 — Perplexity (Training Monitor)

Perplexity is your real-time training health signal. Lower = better. It measures how surprised the model is by held-out text.

```
perplexity = exp(average cross-entropy loss per token)
```

**Implementation in C++:**

```cpp
float compute_perplexity(ModernLLM& model, 
                         const std::vector<Batch>& eval_batches) {
    float total_loss = 0.f;
    int   total_tokens = 0;
    
    model.set_eval_mode();  // disable dropout, use deterministic ops
    
    for (auto& batch : eval_batches) {
        float loss = model.forward(batch);  // no backward
        total_loss += loss * batch.n_tokens;
        total_tokens += batch.n_tokens;
    }
    
    model.set_train_mode();
    return expf(total_loss / total_tokens);
}
```

**Eval dataset:** Hold out 50MB of FineWeb-Edu (never seen during training). Compute perplexity every 1000 steps.

**Typical perplexity targets:**
| Model Size | Target Perplexity (WikiText-103) |
|---|---|
| 300M | ~15–20 |
| 1B | ~12–15 |
| 3B | ~10–12 |
| 7B | ~8–10 |

---

## Layer 2 — Generation Quality (Spot Checks)

Every 5,000 steps, generate responses to a fixed set of prompts and log them. Review manually or with a judge model.

```cpp
// Fixed eval prompts — same every time for comparability
const std::vector<std::string> EVAL_PROMPTS = {
    "Explain quantum entanglement in simple terms.",
    "Write a Python function to find the nth Fibonacci number.",
    "What is the capital of Australia, and why was it chosen?",
    "Solve: if 2x + 5 = 17, what is x?",
    "Summarize the causes of World War 1 in 3 sentences."
};

void run_generation_eval(ModernLLM& model, int step) {
    for (auto& prompt : EVAL_PROMPTS) {
        std::string output = model.generate(prompt, 256, 0.1f); // temp=0.1
        log_to_file("eval_generations.jsonl", {step, prompt, output});
    }
}
```

---

## Layer 3 — Benchmark Suite

Run formally at major checkpoints (every 10K steps during training, and at final release).

### Primary Benchmarks

| Benchmark | Measures | Tool | HF Dataset |
|---|---|---|---|
| **MMLU** | General knowledge (57 subjects) | lm-evaluation-harness | `cais/mmlu` |
| **HellaSwag** | Commonsense reasoning | lm-evaluation-harness | `Rowan/hellaswag` |
| **ARC-Challenge** | Science reasoning | lm-evaluation-harness | `allenai/ai2_arc` |
| **TruthfulQA** | Factual accuracy | lm-evaluation-harness | `truthful_qa` |
| **HumanEval** | Code generation | separate eval | `openai/openai_humaneval` |
| **GSM8K** | Math reasoning | lm-evaluation-harness | `openai/gsm8k` |
| **GPQA** | Graduate-level science | lm-evaluation-harness | `Idavidrein/gpqa` |

**Avoid benchmark contamination:** MMLU and HellaSwag may already be in pre-training data. Use them to track regression (did fine-tuning hurt?), not for absolute capability claims. GPQA is harder to contaminate and better for frontier comparisons.

### Running Benchmarks

Use EleutherAI's `lm-evaluation-harness` (Python) against your model's inference server:

```bash
# Your C++ model exposes OpenAI-compatible API on localhost:8080
lm_eval --model openai-completions \
        --model_args model=modernllm,base_url=http://localhost:8080/v1 \
        --tasks mmlu,hellaswag,arc_challenge,truthfulqa_mc2,gsm8k \
        --num_fewshot 5 \
        --output_path results/step_50000/
```

Or write a minimal C++ HTTP server (or Python wrapper around your C++ model) to serve the benchmark API.

### OpenAI-Compatible Inference Server (Python wrapper)

```python
# inference_server.py
# Wraps your C++ binary in an OpenAI-compatible API
from fastapi import FastAPI
import subprocess, json

app = FastAPI()
model_process = subprocess.Popen(["./modernllm_inference", "--checkpoint", "best.bin"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)

@app.post("/v1/completions")
async def complete(request: dict):
    prompt = request["prompt"]
    max_tokens = request.get("max_tokens", 256)
    
    # Send to C++ process via stdin
    model_process.stdin.write(json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode() + b"\n")
    model_process.stdin.flush()
    result = model_process.stdout.readline()
    return json.loads(result)
```

---

## Layer 4 — RL Reward Tracking

During GRPO training, track these per step:

```python
metrics = {
    "reward_mean":     float,   # average reward across completions
    "reward_std":      float,   # spread (higher = more learning signal)
    "reward_max":      float,   # best completion this batch
    "pass_rate":       float,   # fraction correct (math/code)
    "kl_divergence":   float,   # distance from reference model
    "format_reward":   float,   # chain-of-thought format score
    "response_length": float,   # mean completion length
}
```

**Warning signals:**
- `kl_divergence > 0.5` → reduce LR, model drifting too far
- `reward_std → 0` → all completions same quality, no learning signal
- `reward_mean` not improving after 2000 steps → adjust reward function

---

## Evaluation Schedule

| Training Phase | Eval Frequency | What to Check |
|---|---|---|
| Pre-training | Every 1000 steps | Perplexity, generation spot checks |
| Pre-training | Every 10K steps | MMLU, HellaSwag (5-shot) |
| SFT | Every 500 steps | Perplexity, generation quality |
| DPO | Every 500 steps | Win rate vs SFT model |
| GRPO | Every 500 steps | Reward mean/std, pass@1 on math |
| Final | Once | Full benchmark suite |

---

## Tracking Results

Store all eval results in `eval_results.jsonl`:

```jsonl
{"step":10000,"phase":"pretrain","perplexity":18.2,"mmlu_5shot":0.312,"hellaswag":0.534}
{"step":20000,"phase":"pretrain","perplexity":15.1,"mmlu_5shot":0.341,"hellaswag":0.567}
{"step":5000,"phase":"sft","perplexity":9.2,"generation_quality":"see generations/step_5000/"}
```

Include these results in your HuggingFace model card when publishing.

---

## Final Model Card Metrics (for HuggingFace)

At release, report:

```markdown
## Evaluation Results

| Benchmark | Score | Few-shot |
|---|---|---|
| MMLU | 0.XX | 5-shot |
| HellaSwag | 0.XX | 10-shot |
| ARC-Challenge | 0.XX | 25-shot |
| TruthfulQA | 0.XX | 0-shot |
| HumanEval | 0.XX (pass@1) | 0-shot |
| GSM8K | 0.XX | 8-shot CoT |

Validation perplexity (FineWeb-Edu held-out): XX.X
```
