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
    default_data_collator,
)

def main():
    # 1. Configuration from environment variables (claude.md specs)
    model_name = "EleutherAI/pythia-410m"
    dataset_name = "rajtripathi/5M-Songs-Lyrics"  # 5 million song entries
    epochs = int(os.getenv("EPOCHS", "2"))
    batch_size = int(os.getenv("BATCH_SIZE", "16"))  # Reduced from 32 to fit memory
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

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # 2. Load Model and Tokenizer
        print("Loading model and tokenizer...")
        
        # Load tokenizer first (needed for format_prompt function)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f"{data_dir}/model_cache")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully")

        # Load model with BF16 for L40S Ada architecture (per external docs)  
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # BF16 optimal for L40S Ada architecture
            cache_dir=f"{data_dir}/model_cache",
            # Remove device_map="auto" to avoid distributed training issues in single-node setup
            # attn_implementation="flash_attention_2",  # Comment out until flash-attn is available
        )

        # Manually move model to GPU for single-node training
        model = model.to(device)
        print(f"✅ Model loaded on device: {device}")

        # Print model architecture to debug layer naming
        print("Model architecture components:")
        for name, _ in model.named_children():
            print(f"  - {name}")

        # Freeze all layers except the last two (claude.md specs)
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the last two layers - using correct Pythia architecture
        # Pythia models use GPTNeoXForCausalLM with gpt_neox.layers and embed_out
        try:
            # Try different possible layer naming conventions
            if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
                layers = model.gpt_neox.layers
                for param in layers[-2:].parameters():
                    param.requires_grad = True
                print(f"✅ Unfroze last 2 layers from gpt_neox.layers (total layers: {len(layers)})")
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layers = model.transformer.h  
                for param in layers[-2:].parameters():
                    param.requires_grad = True
                print(f"✅ Unfroze last 2 layers from transformer.h (total layers: {len(layers)})")
            else:
                print("⚠️  Could not find transformer layers, unfreezing all parameters")
                for param in model.parameters():
                    param.requires_grad = True

            # Unfreeze output embedding layer
            if hasattr(model, 'embed_out'):
                for param in model.embed_out.parameters():
                    param.requires_grad = True
                print("✅ Unfroze embed_out layer")
            elif hasattr(model, 'lm_head'):
                for param in model.lm_head.parameters():
                    param.requires_grad = True
                print("✅ Unfroze lm_head layer")
            else:
                print("⚠️  Could not find output layer")

        except Exception as e:
            print(f"⚠️  Error unfreezing layers: {e}")
            print("Falling back to unfreezing all parameters")
            for param in model.parameters():
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

    try:
        # 3. Load and Prepare Dataset
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

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

    # Preprocessing function to format the data for music lyrics understanding
    def format_prompt(example):
        # Format for lyrics analysis and generation as per claude.md goals
        # Handle different dataset column formats
        if 'Instruction' in example and 'Label' in example:
            # This dataset format has Instruction and Label columns
            instruction = example.get('Instruction', '')
            label = example.get('Label', '')
            prompt = f"""### Task: {instruction}
### Response:
{label[:1000]}"""  # Limit response length for training efficiency
        else:
            # Original format with song/artist/lyrics
            song_title = example.get('song', example.get('Song', 'Unknown Song'))
            artist = example.get('artist', example.get('Artist', 'Unknown Artist'))
            lyrics = example.get('lyrics', example.get('Lyrics', ''))
            prompt = f"""### Task: Analyze and understand these song lyrics
### Artist: {artist}
### Song: {song_title}
### Lyrics Analysis:
{lyrics[:1000]}"""  # Limit lyrics length for training efficiency
        
        return {"text": prompt + tokenizer.eos_token}

    try:
        # Apply the formatting to train and validation datasets
        print("Processing datasets...")
        processed_train = train_dataset.map(format_prompt)
        processed_val = val_dataset.map(format_prompt)

        # Tokenize the datasets - shorter sequences to save memory
        def tokenize_function(examples):
            # Use shorter max_length to reduce memory usage
            result = tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=512,  # Reduced from 1024 to save memory
                padding="max_length"
            )
            # For causal LM, labels should be identical to input_ids
            result["labels"] = result["input_ids"].copy()
            return result

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
        print(f"Dataset features after tokenization: {tokenized_dataset.features}")
        print(f"First input_ids type: {type(tokenized_dataset[0]['input_ids'])}")
        print(f"First input_ids sample values: {tokenized_dataset[0]['input_ids'][:10]}")
        print(f"Label type: {type(tokenized_dataset[0]['labels'])} with values: {tokenized_dataset[0]['labels'][:10]}")

    except Exception as e:
        print(f"❌ Error processing datasets: {e}")
        raise

    # Custom trainer callback to track training history
    class TrainingHistoryCallback(TrainerCallback):
        def __init__(self):
            self.history = []
        
        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            if logs:
                self.history.append(logs)

    history_callback = TrainingHistoryCallback()

    try:
        # 4. Set Up Trainer with claude.md specifications
        import transformers
        print(f"Transformers version: {transformers.__version__}")

        # Use the correct TrainingArguments for transformers 4.51.3
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/checkpoints",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            logging_steps=10,
            save_steps=500,
            bf16=True,
            # For transformers 4.51.3, use eval_strategy (not evaluation_strategy)
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_val_dataset,
            data_collator=default_data_collator,
            callbacks=[history_callback],
        )


        # 5. Start Fine-Tuning
        print("🚀 Starting fine-tuning on 5M Songs Lyrics dataset...")
        print(f"Target VRAM usage: 45GB/48GB (94% utilization)")
        training_result = trainer.train()
        print("✅ Fine-tuning complete!")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

    try:
        # 6. Save the best model and training artifacts as per claude.md
        trainer.save_model(f"{output_dir}/best_model")
        tokenizer.save_pretrained(f"{output_dir}/best_model")
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

    except Exception as e:
        print(f"❌ Error saving artifacts: {e}")
        raise

    try:
        # 7. Generate demo outputs for conference demonstration
        print("Generating demo outputs...")
        demo_prompts = [
            "### Task: Analyze and understand these song lyrics\n### Artist: The Beatles\n### Song: Yesterday\n### Lyrics Analysis:",
            "### Task: Analyze and understand these song lyrics\n### Artist: Bob Dylan\n### Song: Blowin' in the Wind\n### Lyrics Analysis:",
            "### Task: Analyze and understand these song lyrics\n### Artist: Nirvana\n### Song: Smells Like Teen Spirit\n### Lyrics Analysis:"
        ]

        demo_outputs = []
        for i, prompt in enumerate(demo_prompts):
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = inputs.to(device)
                    
                with torch.no_grad():
                    output = model.generate(
                        **inputs, 
                        max_new_tokens=150, 
                        do_sample=True, 
                        top_k=50, 
                        top_p=0.95,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                demo_outputs.append({"prompt": prompt, "response": generated_text})
                print(f"\n--- Demo Output {i+1} ---")
                print(generated_text[:500] + "...")
            except Exception as e:
                print(f"⚠️  Error generating demo output {i+1}: {e}")
                demo_outputs.append({"prompt": prompt, "response": f"Error: {e}"})

        # Save demo outputs
        with open(f"{output_dir}/demo_outputs/sample_responses.json", "w") as f:
            json.dump(demo_outputs, f, indent=2)

    except Exception as e:
        print(f"❌ Error generating demo outputs: {e}")
        # Don't raise here, demo outputs are optional

    print(f"\n🎵 Music lyrics fine-tuning completed successfully!")
    print(f"📊 Training artifacts saved to {output_dir}/")
    print(f"🎭 Demo outputs saved to {output_dir}/demo_outputs/")

if __name__ == "__main__":
    main()