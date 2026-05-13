# 09 — Sources & References

> All research papers, technical reports, blogs, and tools used in this project.
> Fetched and verified May 2026.

---

## Core Architecture Papers

### Foundational
| Paper | Link | What It Covers |
|---|---|---|
| Attention Is All You Need (2017) | https://arxiv.org/abs/1706.03762 | Original transformer |
| GPT-2 (2019) | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | Decoder-only LLM |
| Scaling Laws (2020) | https://arxiv.org/abs/2001.08361 | How to scale compute/params/data |

### Attention Improvements
| Paper | arXiv | What It Covers |
|---|---|---|
| GQA: Training Generalized Multi-Query Attention | 2305.13245 | Grouped Query Attention |
| Flash Attention | 2205.14135 | Memory-efficient attention kernel |
| Flash Attention 2 | 2307.08691 | Improved FA parallelism |
| Flash Attention 3 | 2407.08608 | H100-optimized FA |
| RoPE | 2104.09864 | Rotary Position Embeddings |
| ALiBi | 2108.12409 | Linear biases (alternative to RoPE) |
| Sliding Window Attention (Mistral) | 2310.06825 | Efficient long-context attention |

### MLA (Multi-Head Latent Attention)
| Resource | Link |
|---|---|
| DeepSeek-V2 Technical Report | https://arxiv.org/abs/2405.04434 |
| DeepSeek-V3 Technical Report | https://arxiv.org/abs/2412.19437 |
| MLA Explained (Towards Data Science) | https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4 |
| MLA Implementation (DeepWiki) | https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.2-multi-head-latent-attention-(mla) |
| MoE-MLA-RoPE Unified Paper | https://arxiv.org/abs/2508.01261 |

### Mixture of Experts
| Paper | arXiv | What It Covers |
|---|---|---|
| Outrageously Large Neural Networks (MoE) | 1701.06538 | Original MoE for NLP |
| GShard | 2006.16668 | MoE scaling to 600B params |
| Switch Transformer | 2101.03961 | Simplified MoE routing |
| DeepSeekMoE | 2401.06066 | Fine-grained expert decomposition |
| DeepSeek-V3 Report §2 | 2412.19437 | Auxiliary-loss-free load balancing |

### FFN Improvements
| Paper | arXiv | What It Covers |
|---|---|---|
| GLU Variants Improve Transformer (SwiGLU) | 2002.05202 | Gated linear units, SwiGLU |
| RMSNorm | 1910.07467 | Simpler normalization, now universal |

### Multi-Token Prediction
| Paper | arXiv | What It Covers |
|---|---|---|
| Better & Faster Large Language Models via Multi-Token Prediction | 2404.19737 | MTP training objective |

### SSM / Beyond Transformers
| Paper | arXiv | What It Covers |
|---|---|---|
| Mamba | 2312.00752 | Selective state space model |
| Mamba-2 | 2405.21060 | Improved Mamba architecture |
| xLSTM | 2405.04517 | Extended LSTM as LLM alternative |
| Titans | 2501.00663 | Memory-augmented attention |

---

## Training Efficiency Papers

| Paper | arXiv / Link | What It Covers |
|---|---|---|
| Mixed Precision Training | 1710.03740 | FP16/BF16 training |
| Gradient Checkpointing | 1604.06174 | Memory-efficient training |
| AdamW | 1711.05101 | Decoupled weight decay |
| 8-bit Optimizers | 2110.02861 | INT8 optimizer states |
| Muon Optimizer | https://github.com/KellerJordan/Muon | Newton-Schulz orthogonalization |
| Efficient Scaling (GQA, inference) | 2211.05100 | KV cache optimization |

---

## RL Post-Training Papers

| Paper | arXiv | What It Covers |
|---|---|---|
| RLHF (InstructGPT) | 2203.02155 | Original RLHF pipeline |
| DPO | 2305.18290 | Direct Preference Optimization |
| GRPO (DeepSeekMath) | 2402.03300 | Group Relative Policy Optimization |
| DeepSeek-R1 | 2501.12948 | GRPO + verifiable rewards at scale |
| DAPO | 2503.14476 | Improved GRPO with decoupled clipping |
| Online vs Offline RL for LLMs | 2506.21495 | Survey of RL methods |

---

## Data & Datasets

| Resource | Link |
|---|---|
| FineWeb-Edu | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |
| FineWeb-2 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| Dolma | https://huggingface.co/datasets/allenai/dolma |
| The Stack v2 | https://huggingface.co/datasets/bigcode/the-stack-v2 |
| FineMath | https://huggingface.co/datasets/HuggingFaceTB/finemath |
| OpenWebMath | https://huggingface.co/datasets/open-web-math/open-web-math |
| SmolTalk (SFT) | https://huggingface.co/datasets/HuggingFaceTB/smoltalk |
| UltraFeedback (DPO) | https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized |
| NuminaMath (GRPO) | https://huggingface.co/datasets/AI-MO/NuminaMath-CoT |
| Codeforces (GRPO) | https://huggingface.co/datasets/open-r1/codeforces |
| mlabonne/llm-datasets | https://github.com/mlabonne/llm-datasets |
| LLMDataHub | https://github.com/Zjh-819/LLMDataHub |
| HF Streaming Guide | https://huggingface.co/blog/streaming-datasets |

---

## Evaluation

| Resource | Link |
|---|---|
| lm-evaluation-harness (EleutherAI) | https://github.com/EleutherAI/lm-evaluation-harness |
| MMLU | https://huggingface.co/datasets/cais/mmlu |
| HellaSwag | https://huggingface.co/datasets/Rowan/hellaswag |
| GPQA | https://huggingface.co/datasets/Idavidrein/gpqa |
| HumanEval | https://huggingface.co/datasets/openai/openai_humaneval |
| GSM8K | https://huggingface.co/datasets/openai/gsm8k |
| LLM Eval Guide (NVIDIA) | https://developer.nvidia.com/blog/mastering-llm-techniques-evaluation/ |
| LLM Benchmarks Explained | https://www.evidentlyai.com/llm-guide/llm-benchmarks |

---

## Architecture Guides & Blogs

| Resource | Link |
|---|---|
| LLM Architecture Design Guide (MaxPool) | https://maxpool.dev/llm-design/ |
| Transformer Design Guide Pt2 (Rohit Bandaru) | https://rohitbandaru.github.io/blog/Transformer-Design-Guide-Pt2/ |
| Big LLM Architecture Comparison (Sebastian Raschka) | https://magazine.sebastianraschka.com/p/technical-deepseek |
| Three Breakthroughs in Modern Transformer | https://www.eventum.ai/resources/blog/three-breakthroughs-that-shaped-the-modern-transformer-architecture |
| Modern LLM Architectures (Softtech) | https://medium.com/softtechas/advancements-in-modern-llm-architectures-b204fe8f0ee8 |
| Transformer Architecture in 2026 (DEV) | https://dev.to/jintukumardas/transformer-architecture-in-2026-from-attention-to-mixture-of-experts-moe-3d46 |
| GRPO Explained (Cameron Wolfe) | https://cameronrwolfe.substack.com/p/grpo |
| GRPO++ Tricks | https://cameronrwolfe.substack.com/p/grpo-tricks |
| Post-Training in 2026 Survey | https://llm-stats.com/blog/research/post-training-techniques-2026 |
| DPO Guide 2025 (philschmid) | https://www.philschmid.de/rl-with-llms-in-2025-dpo |
| DeepSeek Survey Paper | https://arxiv.org/abs/2507.09955 |
| MLA Inside DeepSeek V3 (Medium) | https://medium.com/@ahabb/inside-deepseek-v3-breaking-down-multi-head-latent-attention-mla-72a71fa5771d |

---

## Publishing Tools

| Tool | Link |
|---|---|
| llama.cpp (GGUF conversion) | https://github.com/ggml-org/llama.cpp |
| safetensors library | https://github.com/huggingface/safetensors |
| HuggingFace Hub API | https://huggingface.co/docs/hub/models-uploading |
| PEFT (LoRA for HF ecosystem) | https://github.com/huggingface/peft |
| TRL (SFT/DPO/GRPO trainer) | https://github.com/huggingface/trl |
| HF Hub GGUF docs | https://huggingface.co/docs/hub/gguf |

---

## Reference Implementations

| Project | Link | Why Relevant |
|---|---|---|
| llama.cpp | https://github.com/ggml-org/llama.cpp | C/C++ LLM inference |
| llama3.c (Karpathy) | https://github.com/karpathy/llama2.c | Simple C LLM implementation |
| nanoGPT (Karpathy) | https://github.com/karpathy/nanoGPT | Clean GPT reference |
| DeepSeek-V3 source | https://github.com/deepseek-ai/DeepSeek-V3 | MLA + MoE reference in Python |
| Modded-NanoGPT | https://github.com/KellerJordan/modded-nanogpt | Muon optimizer, fast training |

---

## C# / Monitoring Tools

| Tool | Link |
|---|---|
| LiveCharts2 (WPF charting) | https://github.com/beto-rodriguez/LiveCharts2 |
| CommunityToolkit.Mvvm | https://github.com/CommunityToolkit/dotnet |
| .NET 8 Named Pipes | https://learn.microsoft.com/en-us/dotnet/standard/io/how-to-use-named-pipes |
