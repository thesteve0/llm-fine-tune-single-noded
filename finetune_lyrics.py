import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# 1. Configuration
model_name = "EleutherAI/pythia-410m"
# next model to try is smolLM from hugging face
# model_name = "HuggingFaceTB/SmolLM3-3B"
dataset_name = "mrYou/Lyrics_eng_dataset"
output_dir = "./pythia-410m-lyrics-finetuned"

# 2. Load Model and Tokenizer
# Load model with 'bfloat16' to save memory if your GPU supports it
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # Use bfloat16 for memory efficiency
    device_map="auto" # Automatically place model on available GPU(s)
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set a padding token if one doesn't exist. GPT-style models usually don't.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Load and Prepare Dataset
# Load a sample of the dataset for this demo.
# This dataset doesn't have splits, so we'll create one.
dataset = load_dataset(dataset_name, split="train").train_test_split(test_size=0.1)
train_dataset = dataset['train']

# Let's inspect the features of the dataset to confirm column names
print("Dataset features:", train_dataset.features)
# Expected output might look like: {'Artist': Value(dtype='string', id=None), 'Song': Value(dtype='string', id=None), 'Lyrics': Value(dtype='string', id=None)}


# Preprocessing function to format the data into a prompt
def format_prompt(example):
    # We will structure the data to teach the model to generate lyrics
    # given an artist and a song title.
    prompt = f"""### Instruction:
Write the lyrics for the song "{example['Song']}" by the artist {example['Artist']}.

### Response:
{example['Lyrics']}"""
    # We add the end-of-sequence token to signal the end of a completion.
    return {"text": prompt + tokenizer.eos_token}

# Apply the formatting to the entire dataset
processed_dataset = train_dataset.map(format_prompt)

# Tokenize the dataset
def tokenize_function(examples):
    # We're tokenizing the 'text' field we created
    # Increased max_length to handle longer song lyrics
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = processed_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names # Remove original columns
)

print(f"Sample from tokenized dataset: {tokenized_dataset[0]}")

# 4. Set Up Trainer
training_args = TrainingArguments(
    output_dir=output_dir,
    #  Maximize GPU utilization with a larger batch size.
    # The L40S's 48GB VRAM can easily handle this for a 410M model.
    per_device_train_batch_size=32,

    # With a large batch size, we don't need gradient accumulation.
    gradient_accumulation_steps=1,

    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,

    #  bf16=True is the optimal setting for the L40S GPU.
    bf16=True,

    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # Data collator handles padding batches dynamically
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 5. Start Fine-Tuning
print("🚀 Starting fine-tuning on lyrics dataset...")
trainer.train()
print("✅ Fine-tuning complete!")

# 6. Save the final model
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")

# 7. Test the fine-tuned model with a simple inference
prompt = """### Instruction:
Write the lyrics for the song "Bohemian Rhapsody" by the artist Queen.

### Response:
"""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# Generate more tokens to get a sense of the lyrics
output = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print("\n--- Inference Test ---")
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("--------------------")