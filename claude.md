# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Primary Goal for this Project:**

I am learning Red Hat OpenShift AI for model training. This project implements **music lyrics understanding fine-tuning** as a compelling demonstration of specialized AI capabilities. The goal is to transform a generic language model into a music expert that can analyze lyrics, understand artistic styles, identify genres, and provide cultural context.

**Project Evolution Context:**
This project evolved from a ResNet-18 CIFAR-10 training project. We successfully implemented single-node PyTorchJob training and added metrics annotations for OpenShift AI monitoring. The next phase focuses on LLM fine-tuning to demonstrate more advanced AI capabilities relevant to modern conference audiences.

**Current Phase: Single-Node LLM Fine-Tuning**
We are implementing single-node fine-tuning first to perfect the demo quality and training pipeline. Future phases will scale to distributed training across multiple nodes to demonstrate performance benefits.

**Target Demo Scenario:**
- **Before Fine-tuning**: Generic model provides basic text analysis ("This appears to be about music and emotions")
- **After Fine-tuning**: Specialized model provides expert-level insights about artist styles, genre characteristics, lyrical themes, and cultural context

**Dataset**: HuggingFace "rajtripathi/5M-Songs-Lyrics" (5 million song entries with lyrics and artists).

**Model from Hugging Face**: "EleutherAI/pythia-410m"

**Training Technique**: FP16 fine tuning with only that last two layers of the network unfrozen. 

**Key Technical Decisions:**
- **No RAG (Retrieval-Augmented Generation)**: Pure fine-tuning approach for reliable demo
- **No External Dependencies**: Self-contained model knowledge
- **Maximum VRAM Utilization**: 45GB/48GB per GPU (94% utilization)
- **No LoRA or Compressions**: We are not using LORA/QLORA or any Quantization

**Red Hat OpenShift AI Cluster Details:**
- **Nodes**: Using 1 for single-node training
- **Per Node Resources**:
    - RAM: 32 GB
    - CPU: 8 vCPU
    - GPU: 48 GB VRAM NVIDIA L40S
- **Total Available**: 96 GB RAM, 24 vCPU, 144 GB VRAM
- **Current Utilization**: 1 node (45 GB VRAM for optimized training)

**Namespace/Project**: lyric-professor (consistent across all phases)

**Storage Configuration:**
- **training-data-pvc**: Repurposed for lyrics dataset storage
- **trained-models-pvc**: Model outputs and checkpoints
- **workspace-pvc**: Working directory and temporary files

### Environment Variables
The training script accepts configuration through environment variables:
- `EPOCHS`: Number of training epochs (default: 2)
- `BATCH_SIZE`: Training batch size per GPU (default: 24)
- `LEARNING_RATE`: Learning rate (default: 2e-4)
- `DATA_DIR`: Directory for lyrics dataset (default: /shared/data)
- `OUTPUT_DIR`: Directory for model outputs (default: /shared/models)

### Output Artifacts (saved to `OUTPUT_DIR`):
- `best_model/`: Best performing model checkpoint
- `training_history.json`: Epoch-by-epoch metrics
- `training_summary.json`: Training configuration and results
- `demo_outputs/`: Sample outputs for demo scenarios

**OpenShift AI Integration:**
- PyTorchJob annotated for metrics collection in OpenShift AI console
- Resource usage monitoring via OpenShift AI → Distributed workloads → Project metrics
- GPU utilization and memory usage tracking

**Training Metrics:**
- Training/validation loss curves
- Learning rate scheduling
- Gradient norms and optimization stability
- Sample throughput and training speed

  