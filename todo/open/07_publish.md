# 07 — Export & Publish

> Export to GGUF + safetensors, publish to HuggingFace, open-source on GitHub.

---

## Export Pipeline

```
C++ checkpoint (.bin)
        ↓
Python export script
        ↓
safetensors (primary HF format) + GGUF (for llama.cpp/Ollama)
        ↓
HuggingFace Hub + GitHub Release
```

---

## Step 1 — Write safetensors Exporter

safetensors is the standard HuggingFace model format. It's safe (no arbitrary code execution) and fast to load.

```python
# export_safetensors.py
import struct, json, numpy as np
from safetensors.numpy import save_file

def load_cpp_checkpoint(path: str) -> dict:
    """Read our custom .bin checkpoint format"""
    weights = {}
    with open(path, "rb") as f:
        # Skip header
        magic = struct.unpack("I", f.read(4))[0]
        assert magic == 0x4C4C4D58
        version, step, tokens_seen = struct.unpack("IQQ", f.read(20))
        loss, lr = struct.unpack("ff", f.read(8))
        config_len = struct.unpack("I", f.read(4))[0]
        config = json.loads(f.read(config_len))
        
        # Read weight tensors
        while True:
            chunk = f.read(4)
            if not chunk: break
            name_len = struct.unpack("I", chunk)[0]
            name = f.read(name_len).decode()
            dtype_byte = struct.unpack("B", f.read(1))[0]
            ndim = struct.unpack("I", f.read(4))[0]
            shape = struct.unpack(f"{ndim}I", f.read(ndim * 4))
            numel = 1
            for s in shape: numel *= s
            dtype = np.float32 if dtype_byte == 0 else np.uint16  # bf16 as uint16
            data = np.frombuffer(f.read(numel * dtype().itemsize), dtype=dtype)
            weights[name] = data.reshape(shape)
    
    return weights, config

def export_safetensors(checkpoint_path: str, output_dir: str):
    weights, config = load_cpp_checkpoint(checkpoint_path)
    
    # Save weights
    save_file(weights, f"{output_dir}/model.safetensors")
    
    # Save config.json (HuggingFace format)
    hf_config = {
        "model_type": "modernllm",
        "hidden_size": config["d_model"],
        "num_hidden_layers": config["n_layers"],
        "num_attention_heads": config["n_heads"],
        "vocab_size": config["vocab_size"],
        "max_position_embeddings": config["max_seq_len"],
        "architectures": ["ModernLLMForCausalLM"],
        "torch_dtype": "bfloat16"
    }
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    
    print(f"Exported to {output_dir}/")

export_safetensors("checkpoints/final.bin", "export/modernllm-300m")
```

---

## Step 2 — Convert to GGUF

GGUF is the format used by llama.cpp and Ollama — this is how people run your model locally without Python.

```bash
# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && pip install -r requirements.txt

# Convert safetensors → GGUF (F16)
python convert_hf_to_gguf.py ../export/modernllm-300m \
    --outtype f16 \
    --outfile ../export/modernllm-300m-f16.gguf

# Quantize to Q4_K_M (most popular format — good quality/size tradeoff)
./llama-quantize ../export/modernllm-300m-f16.gguf \
                 ../export/modernllm-300m-Q4_K_M.gguf Q4_K_M

# Also create Q8_0 (higher quality, larger)
./llama-quantize ../export/modernllm-300m-f16.gguf \
                 ../export/modernllm-300m-Q8_0.gguf Q8_0
```

**Quantization formats to publish:**
| Format | Size | Quality | Use Case |
|---|---|---|---|
| F16 | 600MB (300M) | Lossless | Fine-tuning, research |
| Q8_0 | ~320MB | Near-lossless | High-quality inference |
| Q4_K_M | ~180MB | Good | Daily use, recommended |
| Q3_K_M | ~140MB | Acceptable | Memory-constrained |

---

## Step 3 — Register Custom Architecture on HuggingFace

Since this is a custom architecture (not LLaMA/GPT-2), you need to register it so HF can load it.

```python
# modeling_modernllm.py — HuggingFace-compatible Python implementation
# This is a PYTHON wrapper that loads your safetensors weights
# (not the C++ training code — just for HF ecosystem compatibility)

from transformers import PreTrainedModel, PretrainedConfig
import torch

class ModernLLMConfig(PretrainedConfig):
    model_type = "modernllm"
    def __init__(self, d_model=1024, n_layers=24, n_heads=16, 
                 vocab_size=32000, **kwargs):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        super().__init__(**kwargs)

class ModernLLMForCausalLM(PreTrainedModel):
    config_class = ModernLLMConfig
    # ... forward() implementation in PyTorch
    # This allows: model = ModernLLMForCausalLM.from_pretrained("yourname/modernllm-300m")
```

---

## Step 4 — HuggingFace Upload

```python
# upload_to_hf.py
from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = "YOUR_USERNAME/modernllm-300m"

# Create repo
create_repo(repo_id, repo_type="model", private=False)

# Upload all files
api.upload_folder(
    folder_path="export/modernllm-300m",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Initial release: ModernLLM 300M"
)

# Upload GGUF files
api.upload_file(
    path_or_fileobj="export/modernllm-300m-Q4_K_M.gguf",
    path_in_repo="modernllm-300m-Q4_K_M.gguf",
    repo_id=repo_id,
    repo_type="model"
)
```

---

## Step 5 — Model Card (README.md on HuggingFace)

```markdown
---
license: apache-2.0
base_model: null
tags:
  - custom-architecture
  - mla
  - moe
  - cpp
language:
  - en
---

# ModernLLM-300M

A modern transformer LLM built from scratch in C++, implementing:
- Multi-Head Latent Attention (MLA)
- Mixture of Experts (64 experts, top-8 active)
- RoPE positional encoding
- SwiGLU FFN
- Flash Attention 2
- Multi-Token Prediction training

## Model Details

| | |
|---|---|
| Parameters | 300M total, 37M active |
| Architecture | Decoder-only transformer |
| Context length | 8192 tokens |
| Training tokens | 100B |
| Training data | FineWeb-Edu, The Stack v2, FineMath |
| Post-training | SFT → DPO → GRPO |

## Evaluation

| Benchmark | Score |
|---|---|
| MMLU (5-shot) | X.XX |
| HellaSwag | X.XX |
| GSM8K | X.XX |

## Usage (GGUF / Ollama)
```bash
ollama run hf.co/YOUR_USERNAME/modernllm-300m
```

## Usage (Python)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("YOUR_USERNAME/modernllm-300m",
                                              trust_remote_code=True)
```

## Training Details
Built with a custom C++ training framework. 
See: https://github.com/YOUR_USERNAME/modernllm

## License
Apache 2.0
```

---

## GitHub Repository Structure

```
modernllm/
│
├── README.md                    ← project overview, quick start
├── LICENSE                      ← Apache 2.0
│
├── cpp/                         ← C++ model + trainer
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── core/
│   │   ├── attention/
│   │   ├── ffn/
│   │   ├── model/
│   │   ├── train/
│   │   └── tokenizer/
│   └── tests/
│
├── csharp/                      ← monitoring GUI
│   └── ModernLLM.Monitor/
│
├── scripts/
│   ├── export_safetensors.py    ← checkpoint → safetensors
│   ├── upload_to_hf.py          ← HuggingFace upload
│   ├── inference_server.py      ← OpenAI-compatible API wrapper
│   └── data_server.py           ← HF streaming data sidecar
│
├── modeling/
│   ├── modeling_modernllm.py    ← HF-compatible PyTorch model
│   ├── configuration_modernllm.py
│   └── tokenization_modernllm.py
│
├── configs/
│   ├── modernllm-300m.json
│   ├── modernllm-1b.json
│   └── modernllm-3b.json
│
├── docs/                        ← this plan folder
│   ├── 01_architecture.md
│   ├── 02_training_efficiency.md
│   └── ...
│
└── .github/
    └── workflows/
        └── eval.yml             ← auto-run benchmarks on push
```

---

## GitHub Actions — Auto Eval on Push

```yaml
# .github/workflows/eval.yml
name: Benchmark Evaluation
on:
  release:
    types: [published]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Download model from HF
        run: python scripts/download_model.py
      - name: Run lm-eval-harness
        run: |
          pip install lm-eval
          lm_eval --model hf \
                  --model_args pretrained=YOUR_USERNAME/modernllm-300m \
                  --tasks mmlu,hellaswag,arc_challenge \
                  --output_path results/
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: results/
```
