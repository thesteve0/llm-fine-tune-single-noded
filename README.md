# llm-fine-tune-single-noded

Wilderness Survival & Practical Skills Q&A Fine-tuning with OpenShift AI - Single Node Training

This project demonstrates fine-tuning HuggingFaceTB/SmolLM3-3B using TRL framework on practical skills Q&A data with Red Hat OpenShift AI and PyTorchJob orchestration. The model transforms from a generic language model into a comprehensive wilderness survival expert that provides detailed guidance on essential survival and practical skills.

## Good Questions for Testing

1. How many cups in a gallon?
2. How do I treat a nosebleed?
3. What are the advantages of a mirrorless DSLR camera?
4. What is the easiest loop knot to tie?
5. I have a whistle, what is the right way to signal for help?



## Quick Start

1. **Deploy Training Job**:
   ```bash
   cd deploy
   ./deploy-script.sh
   ```

2. **Monitor Training**:
   ```bash
   kubectl logs -f pythia-finetuning-demo-master-0
   ```

3. **Download Trained Model** (after training completes):

   ```bash
   oc apply -f outputs/temp-pod.yaml
   ```

   Download the model files:
   ```bash
   # Wait for pod to be ready
   oc wait --for=condition=Ready pod/model-extractor

   oc rsync --progress=true model-extractor:/models/ .

   # Clean up
   oc delete pod model-extractor
   ```

## What You'll Get

After training completes, your downloaded model directory will contain:

- `best_model/` - Your fine-tuned SmolLM3-3B wilderness survival expert model
- `training_summary.json` - Training configuration and results
- `training_history.json` - Epoch-by-epoch metrics
- `demo_outputs/sample_responses.json` - Sample Q&A model outputs
- `checkpoints/` - Training checkpoints

## Converting safetensors to gguf: Convert the Model Yourself (The Technical Way)
If you can't find a pre-converted version or want to use your specific safetensors file, you'll need to convert it to GGUF 
yourself using a tool like llama.cpp. This process is more involved but is a great skill to learn.

### Clone llama.cpp: 

First, you need to get the conversion tools from the llama.cpp project on GitHub.

```Bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```
### Install Dependencies: 

Install the required Python packages. It's best to do this in a virtual environment.

```Bash
pip install -r requirements.txt
```

**Expected Difference**: The fine-tuned model provides comprehensive, step-by-step wilderness survival guidance with safety warnings and multiple approaches, while the original model gives more general responses.

The convert.py script in the llama.cpp directory can handle many model types. You'll point it 
to the directory containing your original Pythia safetensors files.

```Bash
python convert_hf_to_gguf.py /path/to/your/pythia/model --outfile lyrics-pythia.gguf --outtype f16
```
Replace `/path/to/your/original/pythia/model/` with the actual path.

`--outtype f16` creates a 16-bit float model, which is a good balance of size and quality.

## Dataset Information

**Source**: [cahlen/offline-practical-skills-qa-synthetic](https://huggingface.co/datasets/cahlen/offline-practical-skills-qa-synthetic)

**Dataset Details**:
- **Size**: 2,845 Q&A pairs
- **Format**: Converted from JSONL to Parquet for efficient loading
- **Topics**: Car maintenance, home repairs, computer troubleshooting, practical life skills
- **Integration**: Dataset embedded directly in Docker container (no runtime download)
- **Runtime Processing**: Converted to TRL chat template format during training

**Sample Q&A**:
```
Question: "For Simple Car Maintenance Checks, What is the recommended tire pressure for my car?"
Answer: "Check your car's owner's manual or the tire information placard on the driver's side doorjamb for the recommended tire pressure."
```

**Data Processing Pipeline**:
1. Downloaded original JSONL format from HuggingFace
2. Transformed to combine topic and question: "For [topic], [question]"
3. Converted to Parquet format for efficient loading
4. Embedded in container for offline training

## Project Details

See [claude.md](claude.md) for complete project documentation, technical specifications, L40S GPU optimization details, and OpenShift AI integration information.