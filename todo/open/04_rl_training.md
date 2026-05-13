# 04 — RL Post-Training Pipeline

> After pre-training, we do three stages: SFT → DPO → GRPO.
> All RL training data is streamed — nothing downloaded.

---

## Overview

```
Pre-trained base model
        ↓
Stage 1: SFT (Supervised Fine-Tuning)
        ↓
Stage 2: DPO (Direct Preference Optimization)
        ↓
Stage 3: GRPO (Group Relative Policy Optimization)
        ↓
Aligned, instruction-following chat model
```

The 2025–2026 SOTA stack has moved away from classical PPO-based RLHF. The current dominant pipeline is SFT → DPO → GRPO with verifiable rewards. This is what DeepSeek-R1, Nemotron, and most frontier open models now use.

---

## Stage 1 — Supervised Fine-Tuning (SFT)

### Goal

Teach the base model to follow instructions and chat in the expected format (system prompt, user turn, assistant turn).

### Data (all streamable from HuggingFace)

| Dataset | HF ID | Size | Why |
|---|---|---|---|
| **SmolTalk** | `HuggingFaceTB/smoltalk` | 1M samples | High quality SFT mix, HF-curated |
| **Magpie-Pro** | `Magpie-Align/Magpie-Pro-1M-v0.1` | 1M samples | Synthetic instruction pairs, excellent diversity |
| **WildChat** | `allenai/WildChat-1M` | 1M turns | Real-world multi-turn conversations |
| **LMSYS-Chat-1M** | `lmsys/lmsys-chat-1m` | 1M samples | Real conversations with 25 LLMs |
| **OpenHermes-2.5** | `teknium/OpenHermes-2.5` | 1M samples | Diverse reasoning, code, math |
| **UltraChat-200k** | `HuggingFaceH4/ultrachat_200k` | 200K turns | Multi-turn, high quality, GPT-4 generated |

**SFT Mix (recommended):**
- 40% SmolTalk (general instruction following)
- 20% Magpie-Pro (diverse tasks)
- 20% OpenHermes-2.5 (reasoning + code)
- 10% WildChat (real user queries)
- 10% UltraChat (multi-turn)

### Format

```
<|system|>
You are a helpful, accurate, and concise assistant.
<|user|>
{instruction}
<|assistant|>
{response}
<|end|>
```

### Training

- Use a **much lower LR** than pre-training: 1e-5 to 5e-5
- Train for 1–3 epochs (SFT can overfit quickly)
- Only compute loss on **assistant tokens** (mask system + user tokens)
- Apply gradient checkpointing — same as pre-training

---

## Stage 2 — DPO (Direct Preference Optimization)

### Goal

Align the model to prefer good responses over bad ones, without needing a separate reward model.

DPO treats alignment as a classification problem: given a prompt and two responses (chosen vs rejected), optimize the model to prefer the chosen one.

### Why DPO over PPO

- No separate reward model needed (simpler, less memory)
- More stable than PPO (no reward hacking)
- Comparable or better quality than PPO for instruction following

### Data (streamable)

| Dataset | HF ID | Size | Why |
|---|---|---|---|
| **UltraFeedback Binarized** | `HuggingFaceH4/ultrafeedback_binarized` | 64K pairs | GPT-4 scored, high quality chosen/rejected |
| **HelpSteer2** | `nvidia/HelpSteer2` | 20K pairs | NVIDIA-curated, helpfulness + safety |
| **Preference-700K** | `hendrydong/preference_700K` | 700K pairs | Large-scale mix of preference data |
| **Argilla DPO Mix** | `argilla/dpo-mix-7k` | 7K pairs | Curated high-quality pairs |

### DPO Loss

```
L_DPO = -E [ log σ(β * (log π(y_w|x) - log π_ref(y_w|x)) 
                    - β * (log π(y_l|x) - log π_ref(y_l|x))) ]
```

Where:
- `y_w` = chosen (winning) response
- `y_l` = rejected (losing) response  
- `π` = current policy (our model)
- `π_ref` = reference policy (SFT model, frozen)
- `β` = temperature (0.1–0.5, controls deviation from reference)

### Implementation in C++

```cpp
struct DPOBatch {
    std::vector<int> prompt_ids;
    std::vector<int> chosen_ids;
    std::vector<int> rejected_ids;
};

float dpo_loss(ModernLLM& policy, ModernLLM& reference,
               const DPOBatch& batch, float beta) {
    
    float log_prob_chosen_policy   = get_log_probs(policy,    batch.prompt_ids, batch.chosen_ids);
    float log_prob_rejected_policy = get_log_probs(policy,    batch.prompt_ids, batch.rejected_ids);
    float log_prob_chosen_ref      = get_log_probs(reference, batch.prompt_ids, batch.chosen_ids);
    float log_prob_rejected_ref    = get_log_probs(reference, batch.prompt_ids, batch.rejected_ids);
    
    float chosen_reward   = beta * (log_prob_chosen_policy   - log_prob_chosen_ref);
    float rejected_reward = beta * (log_prob_rejected_policy - log_prob_rejected_ref);
    
    return -log_sigmoid(chosen_reward - rejected_reward);
}
```

Reference model (SFT checkpoint) is frozen — no gradients, just inference.

---

## Stage 3 — GRPO (Group Relative Policy Optimization)

### Goal

Improve reasoning capabilities using verifiable rewards — math, code, logic. This is what DeepSeek-R1 used to dramatically improve reasoning without human preference labels.

### Why GRPO

GRPO samples **multiple completions per prompt**, computes their rewards, and uses relative comparison within the group as the advantage signal. No value network needed (unlike PPO).

```
For each prompt x:
  Sample G completions: {y_1, y_2, ..., y_G}
  Compute reward r_i for each completion
  Advantage: A_i = (r_i - mean(r)) / std(r)
  Policy gradient update using A_i
```

### GRPO Loss

```
L_GRPO = -E_g [ A_g * log π(y_g|x) ] + β * KL(π || π_ref)
```

The KL term prevents the model from deviating too far from the reference (SFT) model.

### Verifiable Reward Datasets (streamable)

These have objectively correct answers — no reward model needed.

| Dataset | HF ID | Domain | Verification |
|---|---|---|---|
| **NuminaMath** | `AI-MO/NuminaMath-CoT` | Math | Answer correctness check |
| **MATH** | `hendrycks/competition_mathematics` | Competition math | Exact match |
| **GSM8K** | `openai/gsm8k` | Grade school math | Exact number match |
| **HumanEval** | `openai/openai_humaneval` | Code | Unit test execution |
| **MBPP** | `google-research-datasets/mbpp` | Code | Test suite execution |
| **Codeforces** | `open-r1/codeforces` | Competitive programming | Judge execution |
| **LogicBench** | `Mihir-Chauhan/LogicBench` | Logic reasoning | Answer correctness |
| **ARC-Challenge** | `allenai/ai2_arc` | Science MCQ | Exact match |

### Reward Functions

```cpp
// Math reward: check if final answer matches ground truth
float math_reward(const std::string& completion, 
                  const std::string& ground_truth) {
    // Extract final answer (usually in \boxed{} or after "= ")
    std::string extracted = extract_answer(completion);
    if (extracted == ground_truth) return 1.0f;
    
    // Partial credit for correct approach (optional)
    // Check if reasoning steps are valid
    
    return 0.0f;
}

// Code reward: compile + run test suite
float code_reward(const std::string& code,
                  const std::vector<std::string>& test_cases) {
    // Write code to temp file
    // Compile with g++/python
    // Run each test case
    // Return pass@1 = fraction of tests passed
    int passed = 0;
    for (auto& test : test_cases) {
        if (run_test(code, test)) passed++;
    }
    return (float)passed / test_cases.size();
}
```

### Format Reward

Also reward the model for using proper reasoning format:

```cpp
float format_reward(const std::string& completion) {
    // Reward chain-of-thought structure:
    // <think>...</think><answer>...</answer>
    bool has_think  = completion.find("<think>")  != std::string::npos;
    bool has_answer = completion.find("<answer>") != std::string::npos;
    return (has_think && has_answer) ? 0.2f : 0.0f;
}

// Combined reward:
float total_reward = 0.8f * task_reward + 0.2f * format_reward;
```

### GRPO Training Config

```json
{
  "n_completions_per_prompt": 8,
  "max_new_tokens": 512,
  "temperature": 0.8,
  "beta_kl": 0.04,
  "lr": 1e-6,
  "batch_size": 4,
  "accum_steps": 8,
  "max_steps": 10000
}
```

---

## Combined Post-Training Timeline

| Stage | Steps | Data | LR | Duration |
|---|---|---|---|---|
| SFT | 5,000–10,000 | ~1M instruction pairs | 2e-5 | ~1 day |
| DPO | 2,000–5,000 | ~64K preference pairs | 5e-6 | ~12h |
| GRPO | 5,000–20,000 | Verifiable math/code | 1e-6 | ~2 days |

---

## Checkpoint Strategy for RL Stages

- Save checkpoint every 500 steps (RL training can be unstable)
- Keep the last 5 checkpoints + best by eval reward
- If reward collapses (training instability), rollback to last stable checkpoint
- Monitor KL divergence from reference — if it spikes >0.5, reduce LR
