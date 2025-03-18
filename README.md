# LLM Optimization Techniques

Large Language Models (LLMs) require significant computational resources for both training and inference. In this repo we will present and implmenet (WIP!!) a structured taxonomy of optimization techniques to make LLMs more efficient, faster, and less resource-intensive. Idea is to not have comprehensive overview of **all** methods but have major techniques used to either implement them from scratch or show how to use them so we can build efficent LLM and serve it in 2025!

The techniques are divided into three main categories: 
1. **Model-Level**, 
2. **System-Level**
3. **Process-Level Optimizations**.

---

## 1. Model-Level Optimizations

These techniques focus on modifying the model itself, including its architecture, numerical representation, and internal mechanisms, to improve efficiency.

### 1.1 Architecture Optimizations
- **Attention Mechanism Optimizations**:
  - Multi-Query Attention (MQA)
  - Grouped-Query Attention (GQA)
  - Sliding Window/Local Attention
  - Sparse Attention Patterns (e.g., Longformer, BigBird)
  - Linear Attention Mechanisms (e.g., Performer, Reformer)
  - Flash Attention (GPU kernel optimization)
- **Model Structure Optimization**:
  - Mixture of Experts (MoE)
  - Recurrent Memory Mechanisms (e.g., Transformer-XL, Compressive Transformers)

### 1.2 Numerical Precision Optimizations
- **Data Types**:
  - FP32, FP16, BF16, FP8 (merging in new hardware like NVIDIA H100)
- **Quantization Techniques**:
  - Post-Training Quantization (PTQ):
    - INT8/INT4 Quantization
    - GPTQ, AWQ
  - Quantization-Aware Training (QAT)
  - BitsAndBytes Library
- **Sparsity**:
  - Unstructured Sparsity
  - Structured Sparsity (e.g., N:M Sparsity)
  - Pruning (e.g., Lottery Ticket Hypothesis)

---

## 2. System-Level Optimizations

These techniques focus on optimizing the infrastructure, hardware, and software systems used to train and serve LLMs.

### 2.1 Parallelization Strategies
- **Model Parallelism**:
  - Tensor Parallelism
  - Pipeline Parallelism
  - Sequence Parallelism
- **Data Parallelism**:
  - Standard Data Parallelism
  - ZeRO (Zero Redundancy Optimizer)
  - Fully Sharded Data Parallelism (FSDP)
- **Hybrid Parallelism**:
  - Sharded Data Parallelism

### 2.2 Serving and Inference Optimizations
- **Inference Pipeline Optimization**:
  - Prefill Phase Optimization (critical for latency)
  - Decode Phase Optimization (e.g., FlashDecoding) (important for throughput)
- **KV-Cache Management**:
  - KV-Cache Implementation
  - KV-Cache Pruning
  - KV-Cache Quantization
  - KV-Cache Sharing
- **Batching Strategies**:
  - Static Batching
  - Dynamic Batching
  - Continuous Batching
- **Decoding Strategies**:
  - Greedy Decoding
  - Beam Search
  - Speculative Decoding (e.g., Draft Models, Medusa)
  - Nucleus Sampling (Top-p Sampling)
  - Temperature Scaling

### 2.4 Distributed Systems
- **Distributed Training**:
  - ZeRO-Infinity
  - Gradient Checkpointing
- **Distributed Inference**:
  - Model Sharding
  - Request Prioritization

---

## 3. Process-Level Optimizations

These techniques focus on improving the processes involved in training, fine-tuning, and adapting LLMs.

### 3.1 Training Optimizations
- **Efficient Training Techniques**:
  - Mixed Precision Training
  - Gradient Accumulation
  - Gradient Checkpointing
- **Knowledge Transfer**:
  - Knowledge Distillation:
    - Response-Based Distillation
    - Feature-Based Distillation
    - Self-Distillation
  - Pretraining on Smaller Models

### 3.2 Fine-Tuning Optimizations

Parameter-Efficient Fine-Tuning (PEFT):

- LoRA (Low-Rank Adaptation)

- QLoRA (Quantized LoRA)

- Prefix Tuning/P-Tuning

- Adapters

---


### References

1. Comprenhensive survey of LLMS [arxiv](https://arxiv.org/pdf/2312.03863)
2. LLM a survey [arxiv](https://arxiv.org/pdf/2402.06196)
3. TransMLA (MHA Vs GQA Vs MLA) [arxiv](https://arxiv.org/pdf/2502.07864)