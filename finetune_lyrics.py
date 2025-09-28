import torch
import os
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig

def main():
    # Configuration from environment variables
    model_name = "HuggingFaceTB/SmolLM3-3B"
    dataset_path = "data/qa_dataset.parquet"
    epochs = int(os.getenv("EPOCHS", "4"))
    batch_size = int(os.getenv("BATCH_SIZE", "12"))
    learning_rate = float(os.getenv("LEARNING_RATE", "5e-5"))
    data_dir = os.getenv("DATA_DIR", "/shared/data")
    output_dir = os.getenv("OUTPUT_DIR", "/shared/models")

    # Create output directories
    os.makedirs(f"{output_dir}/best_model", exist_ok=True)
    os.makedirs(f"{output_dir}/demo_outputs", exist_ok=True)

    print(f"Configuration:")
    print(f"- Model: {model_name}")
    print(f"- Dataset: {dataset_path}")
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
        # Load Model and Tokenizer
        print("Loading model and tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp/model_cache")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("Tokenizer loaded successfully")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir="/tmp/model_cache",
            attn_implementation="flash_attention_2",
        )

        model = model.to(device)
        print(f"Model loaded on device: {device}")

        # Monitor VRAM usage after model loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"VRAM after model loading: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            print(f"L40S VRAM utilization: {reserved/48*100:.1f}% of 48GB")

        # Freeze all layers except the last ones for memory efficiency
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last 2 layers and output layer
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layers = model.model.layers
                for param in layers[-2:].parameters():
                    param.requires_grad = True
                print(f"Unfroze last 2 layers from model.layers (total layers: {len(layers)})")
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                layers = model.transformer.h
                for param in layers[-2:].parameters():
                    param.requires_grad = True
                print(f"Unfroze last 2 layers from transformer.h (total layers: {len(layers)})")
            elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
                layers = model.gpt_neox.layers
                for param in layers[-2:].parameters():
                    param.requires_grad = True
                print(f"Unfroze last 2 layers from gpt_neox.layers (total layers: {len(layers)})")
            else:
                print("Could not find transformer layers, unfreezing all parameters")
                for param in model.parameters():
                    param.requires_grad = True

            # Unfreeze output embedding layer
            if hasattr(model, 'embed_out'):
                for param in model.embed_out.parameters():
                    param.requires_grad = True
                print("Unfroze embed_out layer")
            elif hasattr(model, 'lm_head'):
                for param in model.lm_head.parameters():
                    param.requires_grad = True
                print("Unfroze lm_head layer")
            else:
                print("Could not find output layer")

        except Exception as e:
            print(f"Error unfreezing layers: {e}")
            print("Falling back to unfreezing all parameters")
            for param in model.parameters():
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    try:
        # Load and Prepare Dataset
        print(f"Loading Q&A dataset from: {dataset_path}")

        dataset = load_dataset("parquet", data_files=dataset_path)
        full_dataset = dataset["train"]

        print(f"Total dataset size: {len(full_dataset):,} Q&A pairs")

        split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']

        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(val_dataset):,}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    # Define wilderness survival expert system prompt
    WILDERNESS_EXPERT_SYSTEM_PROMPT = """You are a wilderness survival and practical skills expert. Your mission is to provide comprehensive, detailed guidance on essential survival and practical skills. Give thorough, step-by-step instructions with explanations of why each step matters.

Your expertise covers:
- Wilderness Survival Basics: Rule of 3s (3 minutes without air, 3 hours without shelter in harsh conditions, 3 days without water, 3 weeks without food), emergency signaling techniques, essential knots, identifying poisonous plants and safe alternatives
- Basic First Aid: Treatment for cuts, burns, sprains, shock, and emergency care procedures
- Simple Car Maintenance: Checking fluids (oil, coolant, brake, transmission), tire inspection and pressure, lights and electrical systems
- Basic Cooking Techniques: Food safety, preparation methods, cooking over open fires, food preservation
- Common Measurement Conversions: Imperial to metric, cooking measurements, distance and weight conversions
- Essential Knots: Bowline, clove hitch, trucker's hitch, figure-eight, sheet bend, and their practical applications

Always provide detailed explanations, safety warnings when relevant, and multiple approaches when possible. Your responses should be comprehensive enough to help someone learn and apply these skills safely and effectively. Aim for thorough, educational responses rather than brief answers."""

    # Preprocessing function to format the Q&A data for TRL SFTTrainer
    def format_prompt(example):
        question = example['full-question']
        answer = example['answer']

        messages = [
            {"role": "system", "content": WILDERNESS_EXPERT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

        return {"messages": messages}

    try:
        # Apply the formatting to train and validation datasets for TRL
        print("Processing datasets for TRL SFTTrainer...")
        processed_train = train_dataset.map(format_prompt)
        processed_val = val_dataset.map(format_prompt)

        print(f"Training samples: {len(processed_train):,}")
        print(f"Validation samples: {len(processed_val):,}")

    except Exception as e:
        print(f"Error processing datasets: {e}")
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
        # Set Up SFTTrainer with TRL for chat model fine-tuning
        import transformers
        import trl
        print(f"Transformers version: {transformers.__version__}")
        print(f"TRL version: {trl.__version__}")

        training_args = SFTConfig(
            output_dir=f"{output_dir}/checkpoints",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=5,
            learning_rate=learning_rate,
            max_grad_norm=1.0,
            num_train_epochs=epochs,
            logging_steps=10,
            save_steps=500,
            bf16=True,
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
            gradient_checkpointing=True,
            dataloader_drop_last=True,
            max_seq_length=1024,
            packing=False,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_train,
            eval_dataset=processed_val,
            callbacks=[history_callback],
        )

        # Start Fine-Tuning
        print("Starting fine-tuning on Q&A dataset with SmolLM3-3B...")
        print(f"Target VRAM usage: <45GB/48GB (L40S GPU capacity)")

        # Pre-training VRAM check
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"VRAM before training: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            if reserved > 40:
                print("WARNING: High VRAM usage before training - risk of OOM!")

        training_result = trainer.train()
        print("Fine-tuning complete!")

        # Post-training VRAM check
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"VRAM after training: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            print(f"Peak L40S VRAM utilization: {reserved/48*100:.1f}% of 48GB")

    except Exception as e:
        print(f"Error during training: {e}")
        raise

    try:
        # Save the best model and training artifacts
        trainer.save_model(f"{output_dir}/best_model")
        tokenizer.save_pretrained(f"{output_dir}/best_model")
        print(f"Best model saved to {output_dir}/best_model")

        # Save training history as JSON
        with open(f"{output_dir}/training_history.json", "w") as f:
            json.dump(history_callback.history, f, indent=2)

        # Save training summary
        training_summary = {
            "model_name": model_name,
            "dataset_path": dataset_path,
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
        print(f"Error saving artifacts: {e}")
        raise

    try:
        # Generate demo outputs for Q&A demonstration
        print("Generating demo outputs...")
        demo_questions = [
            "For Simple Car Maintenance Checks, How often should I check my oil level?",
            "For Basic Home Repairs, How do I fix a leaky faucet?",
            "For Computer Troubleshooting, What should I do if my computer won't start?"
        ]

        # Disable gradient checkpointing for inference
        model.config.use_cache = True
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_disable()

        demo_outputs = []
        for i, question in enumerate(demo_questions):
            try:
                messages = [
                    {"role": "system", "content": WILDERNESS_EXPERT_SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ]

                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = inputs.to(device)

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        top_k=40,
                        top_p=0.9,
                        temperature=0.3,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                demo_outputs.append({"prompt": prompt, "response": generated_text})
                print(f"\n--- Demo Output {i+1} ---")
                print(generated_text[:500] + "...")
            except Exception as e:
                print(f"Error generating demo output {i+1}: {e}")
                demo_outputs.append({"prompt": prompt, "response": f"Error: {e}"})

        # Save demo outputs
        with open(f"{output_dir}/demo_outputs/sample_responses.json", "w") as f:
            json.dump(demo_outputs, f, indent=2)

    except Exception as e:
        print(f"Error generating demo outputs: {e}")
        # Don't raise here, demo outputs are optional

    print(f"\nQ&A fine-tuning completed successfully!")
    print(f"Training artifacts saved to {output_dir}/")
    print(f"Demo outputs saved to {output_dir}/demo_outputs/")

if __name__ == "__main__":
    main()