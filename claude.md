# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Primary Goal for this Project:**

I am learning Red Hat OpenShift AI for model training. This project implements **wilderness survival and practical skills Q&A fine-tuning** as a compelling demonstration of specialized AI capabilities. The goal is to transform a generic language model into a wilderness survival expert that can provide comprehensive, detailed guidance on essential survival and practical skills.

The main goal of this code is to show how trivial it is to take the code in this repo and turn it into a distributed model for Kubeflow.
The distributed tuning is located here:
https://github.com/thesteve0/llm-fine-tune-distributed

**Current Phase - aligning this project to the distributed project**
We are trying to make this code as similar to the distributed fine tuning example BUT it still needs to work. 

**Target Demo Scenario:**
- **Before Fine-tuning**: Generic model provides basic responses to survival questions
- **After Fine-tuning**: Specialized wilderness survival expert provides comprehensive, detailed guidance with step-by-step instructions, safety warnings, and educational explanations

**Dataset**: HuggingFace "cahlen/offline-practical-skills-qa-synthetic" (2,845 practical skills Q&A pairs covering wilderness survival, car maintenance, home repairs, computer troubleshooting, and other practical skills).

**Model from Hugging Face**: "HuggingFaceTB/SmolLM3-3B" (upgraded from Pythia-410m for better performance)

**Training Technique**: Match the distributed technique

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


*Performance numbers with asterisk indicate performance with sparsity optimization

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

**Model Architecture:**
- **Base Model**: SmolLM3-3B (3.075B total parameters)
- **Trainable Parameters**: 418.9M parameters (13.62% of total)
- **Training Strategy**: Last 2 transformer layers + language modeling head
- **Memory Efficient**: Partial layer unfreezing for optimal VRAM usage