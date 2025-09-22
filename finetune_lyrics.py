import torch
import os
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

# 1. Configuration from environment variables (claude.md specs)
model_name = "EleutherAI/pythia-410m"
dataset_name = "rajtripathi/5M-Songs-Lyrics"  # 5 million song entries
epochs = int(os.getenv("EPOCHS", "2"))
batch_size = int(os.getenv("BATCH_SIZE", "24"))
learning_rate = float(os.getenv("LEARNING_RATE", "2e-4"))
data_dir = os.getenv("DATA_DIR", "/shared/data")
output_dir = os.getenv("OUTPUT_DIR", "/shared/models")

# Create output directories as per claude.md specs
os.makedirs(f"{output_dir}/best_model", exist_ok=True)
os.makedirs(f"{output_dir}/demo_outputs", exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

print(f"Configuration:")
print(f"- Model: {model_name}")
print(f"- Dataset: {dataset_name}")
print(f"- Epochs: {epochs}")
print(f"- Batch size: {batch_size}")
print(f"- Learning rate: {learning_rate}")
print(f"- Data dir: {data_dir}")
print(f"- Output dir: {output_dir}")

# 2. Load Model and Tokenizer
# Load model with BF16 for L40S Ada architecture (per external docs)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # BF16 optimal for L40S Ada architecture
    device_map="auto",
    cache_dir=f"{data_dir}/model_cache",
    attn_implementation="flash_attention_2",  # Enable FlashAttention for memory efficiency
)

# Freeze all layers except the last two (claude.md specs)
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the last two layers
for param in model.transformer.h[-2:].parameters():
    param.requires_grad = True
for param in model.lm_head.parameters():
    param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f"{data_dir}/model_cache")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Load and Prepare Dataset
# Load the 5M songs dataset as specified in claude.md
print("Loading 5M Songs Lyrics dataset...")
dataset = load_dataset(dataset_name, cache_dir=f"{data_dir}/dataset_cache")

# Create train/validation split
if "train" in dataset:
    full_dataset = dataset["train"]
else:
    full_dataset = dataset[list(dataset.keys())[0]]  # Use first available split

# Take a subset for efficient training (adjust size based on your needs)
train_size = min(50000, len(full_dataset))  # Limit to 50k for demo
subset_dataset = full_dataset.select(range(train_size))
split_dataset = subset_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

print(f"Dataset features: {train_dataset.features}")
print(f"Training samples: {len(train_dataset):,}")
print(f"Validation samples: {len(val_dataset):,}")

# Preprocessing function to format the data for music lyrics understanding
def format_prompt(example):
    # Format for lyrics analysis and generation as per claude.md goals
    # Check available columns in the dataset
    song_title = example.get('song', example.get('Song', 'Unknown Song'))
    artist = example.get('artist', example.get('Artist', 'Unknown Artist'))
    lyrics = example.get('lyrics', example.get('Lyrics', ''))
    
    prompt = f"""### Task: Analyze and understand these song lyrics
### Artist: {artist}
### Song: {song_title}
### Lyrics Analysis:
{lyrics[:1000]}"""  # Limit lyrics length for training efficiency
    
    return {"text": prompt + tokenizer.eos_token}

# Apply the formatting to train and validation datasets
processed_train = train_dataset.map(format_prompt)
processed_val = val_dataset.map(format_prompt)

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024, padding=False)  # Increase for lyrics context

tokenized_dataset = processed_train.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_val_dataset = processed_val.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names
)

print(f"Sample from tokenized dataset: {tokenized_dataset[0]['input_ids'][:50]}")

# Custom trainer callback to track training history
class TrainingHistoryCallback(TrainerCallback):
    def __init__(self):
        self.history = []
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            self.history.append(logs)

history_callback = TrainingHistoryCallback()

# 4. Set Up Trainer with claude.md specifications
training_args = TrainingArguments(
    output_dir=f"{output_dir}/checkpoints",
    per_device_train_batch_size=batch_size,  # From environment variable
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    learning_rate=learning_rate,  # From environment variable
    num_train_epochs=epochs,  # From environment variable
    logging_steps=10,
    eval_steps=100,
    save_steps=500,
    bf16=True,  # BF16 training for L40S Ada architecture
    gradient_accumulation_steps=4,  # Increase effective batch size for better utilization
    save_strategy="steps",
    evaluation_strategy="steps", 
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,  # Optimize data loading
    remove_unused_columns=False,  # Keep all columns for debugging
    report_to=[],  # Disable wandb/tensorboard for clean demo
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[history_callback],
)

# 5. Start Fine-Tuning
print("🚀 Starting fine-tuning on 5M Songs Lyrics dataset...")
print(f"Target VRAM usage: 45GB/48GB (94% utilization)")
training_result = trainer.train()
print("✅ Fine-tuning complete!")

# 6. Save the best model and training artifacts as per claude.md
trainer.save_model(f"{output_dir}/best_model")
print(f"Best model saved to {output_dir}/best_model")

# Save training history as JSON
with open(f"{output_dir}/training_history.json", "w") as f:
    json.dump(history_callback.history, f, indent=2)

# Save training summary
training_summary = {
    "model_name": model_name,
    "dataset_name": dataset_name,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "trainable_params": trainable_params,
    "total_params": total_params,
    "training_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
    "final_train_loss": training_result.training_loss if hasattr(training_result, 'training_loss') else None
}

with open(f"{output_dir}/training_summary.json", "w") as f:
    json.dump(training_summary, f, indent=2)

# 7. Generate demo outputs for conference demonstration
demo_prompts = [
    "### Task: Analyze and understand these song lyrics\n### Artist: The Beatles\n### Song: Yesterday\n### Lyrics Analysis:",
    "### Task: Analyze and understand these song lyrics\n### Artist: Bob Dylan\n### Song: Blowin' in the Wind\n### Lyrics Analysis:",
    "### Task: Analyze and understand these song lyrics\n### Artist: Nirvana\n### Song: Smells Like Teen Spirit\n### Lyrics Analysis:"
]

demo_outputs = []
for i, prompt in enumerate(demo_prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=150, 
            do_sample=True, 
            top_k=50, 
            top_p=0.95,
            temperature=0.7
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    demo_outputs.append({"prompt": prompt, "response": generated_text})
    print(f"\n--- Demo Output {i+1} ---")
    print(generated_text[:500] + "...")

# Save demo outputs
with open(f"{output_dir}/demo_outputs/sample_responses.json", "w") as f:
    json.dump(demo_outputs, f, indent=2)

print(f"\n🎵 Music lyrics fine-tuning completed successfully!")
print(f"📊 Training artifacts saved to {output_dir}/")
print(f"🎭 Demo outputs saved to {output_dir}/demo_outputs/")