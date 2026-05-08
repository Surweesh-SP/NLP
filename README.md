# 🏌️‍♂️ Parameter Golf: Relaxed Recursive Transformer (RRT)

This repository contains a novel, parameter-efficient transformer architecture designed for the **Parameter Golf** challenge. The model, dubbed the **Relaxed Recursive Transformer (RRT)**, achieves high performance on language modeling tasks while staying within strict parameter and storage budgets (16MB compressed).

## 🚀 Key Features

*   **Recursive Parameter Sharing:** Reuses a single high-capacity transformer block across multiple recurrence steps.
*   **Per-Step LoRA Adapters:** Differentiates recursive steps using lightweight rank-16 adapters for Q, K, V, O, FC, and Projection matrices.
*   **Bigram Hash Embedding:** Supplements standard embeddings with a 10,240-entry bigram hash table for enhanced vocabulary capacity.
*   **SmearGate:** A temporal gating mechanism that improves coherence by mixing the current hidden state with the previous token's state.
*   **U-Net Style Skip Connections:** Caches hidden states from "encoder" steps and injects them back during "decoder" steps.
*   **INT5/INT6 QAT:** Quantization-Aware Training with mixed-precision weights (5-bit MLP, 6-bit Attention) for extreme compression.
*   **Muon Optimizer:** Utilizes the Muon optimizer (Nesterov + decoupled weight decay) for stable and fast convergence.

## 🏗️ Architecture

The model applies a shared transformer block recursively, using unique positional signals and adapters at each step to simulate a deeper network without the linear increase in parameters.

```mermaid
graph TD
    subgraph Input_Stage ["Input Stage"]
        IDS[Token IDs] --> EMB[Token Embedding]
        IDS --> BGR[Bigram Hash]
        EMB --> ADD1((+))
        BGR --> ADD1
        ADD1 --> N0[RMSNorm]
        N0 --> X0[x0: Hidden Highway]
    end

    subgraph Recursive_Engine ["Recursive Engine (8-14 Steps)"]
        X0 --> MIX((Mix))
        X_PREV[x_{s-1}] --> MIX
        
        subgraph Shared_Block ["Shared Block"]
            MIX --> AN[RMSNorm]
            AN --> ATTN[Attention + LoRA]
            ATTN --> SMEAR[SmearGate]
            SMEAR --> ADD2((+))
            ADD2 --> MN[RMSNorm]
            MN --> MLP[MLP + LoRA]
            MLP --> ADD3((+))
        end

        subgraph Skips ["U-Net Skips"]
            ENC_S[Encoder Steps] -.->|Skip| DEC_S[Decoder Steps]
        end

        ADD3 --> X_NEXT[x_s]
        X_NEXT -.->|Loop| X_PREV
    end

    subgraph Output_Stage ["Output Stage"]
        X_NEXT --> ON[RMSNorm]
        ON --> LIN[Tied Head]
        LIN --> OUT[Logits]
    end
```

## 📊 Performance Results

Based on the latest runs on an NVIDIA RTX 3090:

| Metric | Result |
| :--- | :--- |
| **Total Parameters** | 9.41 Million |
| **Final val_loss** | 3.3640 |
| **Final val_bpb** | **1.9923** |
| **Artifact Size** | 11.08 MB (Compressed zstd-22) |
| **Headroom** | 4.9 MB (Under 16MB limit) |

## 🛠️ Usage

### Installation
```bash
pip install torch sentencepiece numpy zstandard
```

### Training
To start training with the default recursive configuration:
```bash
# Windows
set RUN_ID=rrt_v1
python model.py

# Linux/macOS
PYTHONUTF8=1 RUN_ID=rrt_v1 python model.py
```

### Evaluation
The model automatically performs validation every 400 steps, calculating both cross-entropy loss and Bits-Per-Byte (BPB).

## 🎛️ Hyperparameters

Key parameters found in `HP` class:
*   `model_dim`: 512
*   `num_recur_steps`: 8 (as per output.txt) or 14 (config default)
*   `lora_rank`: 16
*   `bigram_size`: 10240
*   `mlp_mult`: 3

## 📜 License
MIT License
