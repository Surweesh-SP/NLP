# ⚡ Relaxed Recursive Transformer (RRT)

### A Parameter-Efficient NLP Architecture for Compression-Aware Language Modeling

***

## Overview

Relaxed Recursive Transformer (RRT) is a compact NLP architecture built for strong language modeling under tight memory and parameter budgets. Instead of stacking many independent transformer layers, RRT reuses a single shared transformer block across multiple recursive steps, then adds lightweight step-specific adaptation to preserve expressiveness.

This design targets a practical question: can depth be approximated through recurrence while keeping the model small enough for constrained hardware? In this implementation, the answer is yes—the model combines recursive computation, low-rank adaptation, and compression-oriented training to reach strong bits-per-byte performance within a 16 MB deployment budget.

## Key Idea

The core idea is simple:

> Replace depth with recurrence.

A conventional transformer grows capacity by adding more layers, which increases parameters linearly with depth. RRT instead applies one shared transformer block repeatedly for multiple refinement steps, allowing the network to behave like a deeper system without paying the full parameter cost of separate layers.

To avoid making every recursive pass identical, each step gets its own LoRA adapters. This creates a useful balance: the heavy base block is shared, while the low-rank adapters let each step specialize.

## Architecture

RRT is designed around a single principle: reuse a strong transformer block recursively instead of stacking many separate layers. This turns architectural depth into iterative computation, allowing the model to refine representations step by step while keeping the parameter count low.

### Recursive Computation Instead of Deep Stacking

Traditional transformers use many distinct layers, each with its own parameters. RRT replaces this pattern with one shared transformer block that is applied repeatedly across multiple recursive steps.

Each pass updates the representation progressively:

- Early steps capture syntax and local token patterns.
- Middle steps align broader context and dependencies.
- Later steps refine semantics and improve prediction quality.

This creates an iterative reasoning pipeline in which the model revisits and improves the same latent representation over time.

### Step-wise Adaptation with LoRA

Because the same block is reused at every step, the model still needs a mechanism for step-specific behavior. RRT achieves this with LoRA (Low-Rank Adaptation) modules attached separately for each recursive step.

This creates a clean division of roles:

- The shared transformer block provides the core computation.
- The step-wise LoRA adapters provide lightweight specialization.

As a result, the architecture gains some of the expressive benefits of deep transformers without duplicating full parameter sets at every layer.

### Hybrid Token Representation

RRT does not rely only on standard token embeddings. Instead, it combines:

- Standard embeddings for semantic token meaning.
- Bigram hash features for local token-pair structure.

This hybrid representation strengthens short-range dependency modeling and acts as a compact inductive bias, which is especially valuable in a parameter-efficient setting.

### Efficient Attention Mechanism

The attention stack is streamlined for compact, high-efficiency language modeling. Key components include:

- Grouped Query Attention (GQA) to reduce attention compute and memory cost.
- Rotary Positional Encoding (RoPE) to preserve sequence order information efficiently.
- QK scaling or gain control to stabilize attention sharpness.

Together, these choices preserve contextual reasoning quality while remaining practical for low-budget deployment.

### SmearGate

A distinctive component of the model is **SmearGate**, which softly blends the current token representation with information from the previous token. This acts as a lightweight temporal smoothing mechanism.

Its role is to:

- Improve continuity across neighboring tokens.
- Provide a soft memory effect beyond attention.
- Stabilize sequence modeling behavior.

### Step Embeddings

Each recursive step is assigned a learnable step embedding so the model knows which iteration it is currently performing. This gives the shared block a notion of implicit depth.

In effect, step embeddings let different recursive passes behave more like different layers in a deep network, even though the main block weights are shared.

### Adaptive Halting

RRT can optionally use adaptive halting to make computation input-dependent. Instead of always running the same number of recursive steps, the model can decide whether additional refinement is necessary.

This enables:

- Fewer steps for easy tokens.
- More steps for difficult or ambiguous tokens.
- Better compute efficiency during inference.

### Compression-Oriented Design

The architecture is tightly aligned with resource-efficient deployment. Its design works hand in hand with:

- Parameter sharing.
- Compact embeddings.
- Low-precision quantization such as INT5 and INT6.

This makes RRT especially suitable for low-memory environments, edge deployment, and parameter-constrained research settings.

### End-to-End Flow

```text
Input Tokens
  ↓
Embedding + Bigram Features
  ↓
Recursive Transformer (N Steps)
  ├── Shared Block
  ├── Step-specific LoRA
  ├── SmearGate + Attention + MLP
  ↓
Final Representation
  ↓
Output Logits → Prediction
```

### NLP Perspective

RRT can be viewed as an iterative language reasoning system that progressively refines meaning using shared knowledge and lightweight step-specific adaptations. Instead of scaling parameters aggressively, it scales computation over time.

This makes the architecture:

- Efficient.
- Scalable under tight model budgets.
- Conceptually aligned with iterative reasoning rather than purely feedforward depth.

## Training Strategy

The training recipe is as important as the architecture.

### Quantization-Aware Training (QAT)

- INT6 for attention paths.
- INT5 for MLP paths.
- Preserves quality under low-precision deployment constraints.

### Stochastic Weight Averaging (SWA)

- Improves generalization.
- Helps stabilize the final solution.
- Often gives smoother validation behavior.

### Muon Optimizer

- Uses orthogonalized update dynamics.
- Encourages efficient optimization.
- Fits well with compact, carefully tuned architectures.

### Gradient Checkpointing

- Reduces memory usage during training.
- Makes deeper recursion feasible.
- Trades additional compute for lower memory pressure.

## Results

The current implementation reports the following results from training logs:

| Metric | Value |
|--------|-------|
| Parameters | 9.41M |
| Validation Loss | 3.364 |
| Bits-per-byte (BPB) | 1.9923 |
| Model Size | 11.04 MB |
| Budget | 16 MB |

These numbers show that the model stays comfortably within the deployment limit while maintaining competitive compression-oriented language modeling performance.

## Why BPB Matters

Bits-per-byte (BPB) is a compression-linked metric that measures how efficiently the model predicts text. Lower BPB means the model assigns higher probability to the observed sequence, which generally indicates better language modeling efficiency.

For compression-aware NLP, BPB is especially meaningful because it aligns model quality with information efficiency rather than just raw parameter count or conventional perplexity reporting.

## Dataset

- Dataset: FineWeb (tokenized)
- Vocabulary: BPE with 1024 tokens
- Storage format: binary shard files (`.bin`)

This setup is lightweight and practical for efficient training pipelines focused on compact language models.

## NLP Interpretation

RRT can be viewed as a recursive information refinement system.

At each step, the model revisits the same representation space and improves it incrementally. Instead of allocating a new layer for every stage of abstraction, it performs repeated transformation with shared structure and step-specific adaptation.

Conceptually, each recursive step helps:

- Refine semantic structure.
- Reduce predictive uncertainty.
- Improve next-token estimation.

## Main Innovations

### Parameter Efficiency

- Shared core weights across recursive steps.
- LoRA adapters replace the need for fully separate deep stacks.
- Strong capacity-to-parameter ratio.

### Compression-Oriented NLP

- Optimized with BPB as a central outcome.
- Suitable for low-memory language modeling.
- Better aligned with compression-aware evaluation.

### Hybrid Modeling

The architecture combines three useful ideas:

- N-gram-style local bias through Bigram Hash Embedding.
- Transformer attention for contextual reasoning.
- Recursive computation for depth through reuse.

## Trade-offs

| Strengths | Limitations |
|-----------|-------------|
| Extremely parameter-efficient | May be weaker on very long-range reasoning |
| Small deployment footprint | Requires careful recursion tuning |
| Strong compression-oriented performance | More complex than a plain transformer baseline |
| Hardware-friendly under tight budgets | Training dynamics can be sensitive |

## System Flow

```text
Input Tokens
  ↓
Embedding + Bigram Features
  ↓
Recursive Transformer Steps
  ↓
Step-wise LoRA Adaptation
  ↓
SmearGate + Attention + MLP
  ↓
Final Representation
  ↓
Logits → Prediction
```

## Why It Matters

RRT is well suited for settings where model quality must be balanced against strict resource limits, such as:

- Edge devices
- Low-memory inference environments
- Efficient NLP research
- Compression-focused model design
- Small-footprint deployment pipelines

This makes it a strong candidate for practical language modeling where every megabyte matters.

## Future Work

Possible extensions include:

- Retrieval-augmented generation (RAG)
- Dynamic token routing
- Multi-task transfer to QA and summarization
- Adaptive step allocation per token
- Further quantized deployment optimization

## Run

```bash
PYTHONUTF8=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RUN_ID=recursive_model \
python model.py
```

## Summary

This project demonstrates that high-quality NLP does not always require very large models. With recurrence, low-rank adaptation, and compression-aware training, a compact architecture can still deliver strong results under strict deployment constraints.

## Author

**Surweesh SP**  
