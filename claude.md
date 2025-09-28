# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Primary Goal for this Project:**

I am learning Red Hat OpenShift AI for model training. This project implements **wilderness survival and practical skills Q&A fine-tuning** as a compelling demonstration of specialized AI capabilities. The goal is to transform a generic language model into a wilderness survival expert that can provide comprehensive, detailed guidance on essential survival and practical skills.

**Project Evolution Context:**
This project evolved from a ResNet-18 CIFAR-10 training project and lyrics fine-tuning demonstration. We successfully implemented single-node PyTorchJob training with OpenShift AI monitoring. The current phase focuses on advanced LLM fine-tuning using TRL framework to create a specialized wilderness survival expert.

**Current Phase: TRL-Based Fine-Tuning**
We have implemented TRL (Transformers Reinforcement Learning) framework for advanced chat model fine-tuning. This approach provides better training methodology for conversational AI and proper chat template handling.

**Target Demo Scenario:**
- **Before Fine-tuning**: Generic model provides basic responses to survival questions
- **After Fine-tuning**: Specialized wilderness survival expert provides comprehensive, detailed guidance with step-by-step instructions, safety warnings, and educational explanations

**Dataset**: HuggingFace "cahlen/offline-practical-skills-qa-synthetic" (2,845 practical skills Q&A pairs covering wilderness survival, car maintenance, home repairs, computer troubleshooting, and other practical skills).

**Model from Hugging Face**: "HuggingFaceTB/SmolLM3-3B" (upgraded from Pythia-410m for better performance)

**Training Technique**: BF16 fine-tuning using TRL SFTTrainer with last layer unfrozen and comprehensive wilderness survival expert system prompt.

## NVIDIA L40S GPU Specifications

The project uses NVIDIA L40S GPUs optimized for AI/ML workloads:

**Core Architecture:**
- **GPU Architecture**: NVIDIA Ada Lovelace architecture
- **CUDA Cores**: 18,176 NVIDIA Ada Lovelace Architecture-Based CUDA® Cores
- **RT Cores**: 142 NVIDIA Third-Generation RT Cores
- **Tensor Cores**: 568 NVIDIA Fourth-Generation Tensor Cores

**Memory:**
- **GPU Memory**: 48GB GDDR6 with ECC
- **Memory Bandwidth**: 864GB/s
- **Interconnect**: PCIe Gen4 x16 (64GB/s bidirectional)

**Performance Specifications:**
- **FP32 Performance**: 91.6 TFLOPS
- **RT Core Performance**: 212 TFLOPS
- **TF32 Tensor**: 183/366* TFLOPS
- **BFLOAT16 Tensor**: 362.05/733* TFLOPS
- **FP16 Tensor**: 362.05/733* TFLOPS
- **FP8 Tensor**: 733/1,466* TFLOPS
- **INT8 Tensor**: 733/1,466* TOPS
- **INT4 Tensor**: 733/1,466* TOPS

**Physical & Power:**
- **Form Factor**: 4.4" (H) x 10.5" (L), dual slot
- **Max Power**: 350W
- **Power Connector**: 16-pin
- **Thermal**: Passive cooling
- **Display**: 4x DisplayPort 1.4a

**AI/ML Features:**
- **Multi-Instance GPU (MIG)**: No
- **NVLink Support**: No
- **vGPU Support**: Yes
- **NVENC/NVDEC**: 3x/3x (includes AV1 encode and decode)

*Performance numbers with asterisk indicate performance with sparsity optimization

**Key Technical Decisions:**
- **TRL Framework**: Using SFTTrainer with SFTConfig for advanced chat model fine-tuning
- **SmolLM3-3B**: Upgraded to larger, more capable model architecture
- **No RAG (Retrieval-Augmented Generation)**: Pure fine-tuning approach for reliable demo
- **BF16 Training**: Optimal precision for L40S Ada Lovelace architecture
- **Maximum VRAM Utilization**: 45GB/48GB per GPU (94% utilization). Still not achieving this but ran out of time
- **Chat Template**: Proper ChatML format with comprehensive system prompts

**Red Hat OpenShift AI Cluster Details:**
- **Nodes**: Using 1 for single-node training
- **Per Node Resources**:
    - RAM: 20 GB (increased for TRL requirements)
    - CPU: 6 vCPU
    - GPU: 48 GB VRAM NVIDIA L40S
    - If you increase above this for the base machine their will be problems finding nodes to schedule the pod on
- **Current Utilization**: 1 node (45 GB VRAM for optimized training)

**Namespace/Project**: lyric-professor (consistent across all phases)

**Dataset Integration:**
- **Source**: Downloaded from HuggingFace `cahlen/offline-practical-skills-qa-synthetic`
- **Format**: Converted from JSONL to Parquet for efficient loading (77.7% size reduction)
- **Schema**: Two columns - "full-question" (format: "For [topic], [question]") and "answer"
- **Containerization**: Dataset embedded directly in Docker container (no runtime download required)
- **Processing**: TRL-compatible message format with comprehensive wilderness survival system prompt

**Storage Configuration:**
- **trained-models-pvc**: Model outputs and checkpoints
- **workspace-pvc**: Working directory and temporary files
- **Data**: Embedded in container (no separate PVC needed)

### Environment Variables
The training script accepts configuration through environment variables:
- `EPOCHS`: Number of training epochs (default: 4)
- `BATCH_SIZE`: Training batch size per GPU (default: 12)
- `LEARNING_RATE`: Learning rate (default: 5e-5, TRL recommended)
- `DATA_DIR`: Directory for dataset (default: /shared/data)
- `OUTPUT_DIR`: Directory for model outputs (default: /shared/models)

### Wilderness Survival Expert System Prompt:

The model is trained with a comprehensive system prompt that establishes it as a wilderness survival and practical skills expert covering:

**Core Expertise Areas:**
- **Wilderness Survival Basics**: Rule of 3s, emergency signaling, essential knots, plant identification
- **Basic First Aid**: Treatment for cuts, burns, sprains, shock, emergency procedures
- **Simple Car Maintenance**: Fluid checks, tire inspection, electrical systems
- **Basic Cooking Techniques**: Food safety, preparation, cooking over fires, preservation
- **Common Measurement Conversions**: Imperial to metric, cooking measurements, distances
- **Essential Knots**: Bowline, clove hitch, trucker's hitch, figure-eight, sheet bend

**Response Guidelines:**
- Provide thorough, step-by-step instructions with explanations
- Include safety warnings when relevant
- Offer multiple approaches when possible
- Educational responses rather than brief answers
- Comprehensive enough for practical application

### Dataset Details:

**Source Information:**
- **Origin**: HuggingFace dataset `cahlen/offline-practical-skills-qa-synthetic`
- **License**: Synthetic dataset for educational/training purposes
- **Size**: 2,845 Q&A pairs (714KB original JSONL → 159KB Parquet)
- **Topics**: Wilderness survival, car maintenance, home repairs, computer troubleshooting, practical life skills

**Data Processing Pipeline:**
1. **Download**: Original JSONL format from HuggingFace
2. **Transform**: Concatenate topic + question into "full-question" field
3. **Convert**: JSONL → Parquet for efficient ML loading
4. **Embed**: Include dataset directly in Docker container
5. **Load**: Use HuggingFace datasets library with local parquet file
6. **Runtime Format**: Convert to TRL chat template format during training via `format_prompt()` function

**Sample Prompt Structure:**
```
messages: [
  {"role": "system", "content": "You are a wilderness survival expert..."},
  {"role": "user", "content": "For Simple Car Maintenance Checks, What is the recommended tire pressure for my car?"},
  {"role": "assistant", "content": "Check your car's owner's manual or the tire information placard on the driver's side doorjamb for the recommended tire pressure."}
]
```

### Output Artifacts (saved to `OUTPUT_DIR`):
- `best_model/`: Best performing model checkpoint with tokenizer
- `training_history.json`: Epoch-by-epoch metrics and logs
- `training_summary.json`: Training configuration and results
- `demo_outputs/`: Sample Q&A outputs for demo scenarios
- `checkpoints/`: Training checkpoints for recovery

### Inference Scripts:
- `ask_tuned_model.py`: Query the fine-tuned wilderness survival expert
- `ask_original_model.py`: Query the original SmolLM3-3B for comparison
- Both scripts use the same comprehensive system prompt for fair comparison

**OpenShift AI Integration:**
- PyTorchJob annotated for metrics collection in OpenShift AI console
- Resource usage monitoring via OpenShift AI → Distributed workloads → Project metrics
- GPU utilization and memory usage tracking
- Updated for wilderness survival training focus

**Training Framework:**
- **TRL SFTTrainer**: Advanced supervised fine-tuning for chat models
- **SFTConfig**: Optimized training arguments for conversational AI
- **Chat Template**: Proper ChatML format handling
- **System Prompt Integration**: Consistent training and inference prompts
- **BF16 Precision**: Optimal for L40S Ada Lovelace architecture

**Training Metrics:**
- Training/validation loss curves
- Learning rate scheduling
- Gradient norms and optimization stability
- Sample throughput and training speed
- Model parameter efficiency (13.62% trainable parameters)

## Performance Optimizations for NVIDIA L40S

**TRL Framework Benefits:**
- **Optimized Chat Training**: Purpose-built for conversational AI fine-tuning
- **Memory Efficiency**: Better memory management compared to basic Trainer
- **Template Handling**: Automatic chat template processing
- **Advanced Features**: Support for advanced training techniques

**L40S-Specific Performance Benefits:**
- **2x Memory Capacity**: 48GB enables larger batch sizes and longer sequences
- **Enhanced Tensor Performance**: 362-733 TFLOPS for BF16 operations
- **Ada Lovelace Architecture**: 4th-gen Tensor Cores with improved AI/ML efficiency
- **High Memory Bandwidth**: 864GB/s for faster data throughput

**Current Optimization Settings:**
- **Batch Size**: 12 per device with gradient accumulation of 5 (effective batch size: 60)
- **Sequence Length**: 1024 tokens for comprehensive responses
- **BF16 Training**: Leveraging L40S Ada Lovelace optimization
- **Gradient Checkpointing**: Enabled for memory efficiency
- **Flash Attention 2**: Optimized attention computation

**Model Architecture:**
- **Base Model**: SmolLM3-3B (3.075B total parameters)
- **Trainable Parameters**: 418.9M parameters (13.62% of total)
- **Training Strategy**: Last 2 transformer layers + language modeling head
- **Memory Efficient**: Partial layer unfreezing for optimal VRAM usage