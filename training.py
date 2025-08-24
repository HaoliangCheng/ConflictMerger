import os
import sys
import json
import argparse
import time
import difflib
import re
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    AutoConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from datasets import Dataset
from trl import SFTTrainer

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
MAX_TOKENS = 32768

def get_conflict_dirs(base_dir):
    """Get all conflict directories in the dataset."""
    conflict_dirs = []
    
    # Check if base_dir exists
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist.")
        return []
    
    # List all subdirectories in the dataset directory
    for conflict_dir in os.listdir(base_dir):
        conflict_path = os.path.join(base_dir, conflict_dir)
        
        # Skip non-directories and hidden files
        if not os.path.isdir(conflict_path) or conflict_dir.startswith('.'):
            continue
            
        # Check if the directory contains merged.py and M.py
        if (os.path.exists(os.path.join(conflict_path, 'merged.py')) and 
            os.path.exists(os.path.join(conflict_path, 'M.py'))):
            conflict_dirs.append(conflict_path)
    
    return conflict_dirs

def read_file_content(file_path):
    """Read the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def calculate_similarity(file1_content, file2_content):
    """Calculate similarity between two files using difflib."""
    if not file1_content or not file2_content:
        return 0.0
    
    matcher = difflib.SequenceMatcher(None, file1_content, file2_content)
    return matcher.ratio()

def analyze_conflict(conflict_dir, model_name, prompt, use_trained_model=False, peft_model_id=None):
    """Analyze a single conflict and return resolution results."""
    conflict_name = os.path.basename(conflict_dir)
    print(f"\nAnalyzing conflict: {conflict_name}")

    # Check for the expected files
    merged_path = os.path.join(conflict_dir, 'merged.py')
    m_path = os.path.join(conflict_dir, 'M.py')

    # Read file contents
    merged_content = read_file_content(merged_path)
    m_content = read_file_content(m_path)

    if not merged_content or not m_content:
        print(f"  Error: Could not read required files for {conflict_name}")
        return {
            'conflict_dir': conflict_name,
            'similarity': 0,
            'inference_time': 0,
            'resolved_code': "None"
        }

    # Load model
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Load fine-tuned model if specified
    if use_trained_model and peft_model_id:
        model = PeftModel.from_pretrained(model, peft_model_id)

    # Prepare messages
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": merged_content}
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    text = tokenizer.apply_chat_template(
       messages,
       tokenize=False,
       add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    start_time = time.time()

    generated_ids = model.generate(
      **model_inputs,
       max_new_tokens=MAX_TOKENS
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"  Resolution completed in {inference_time:.2f} seconds")

    # Clean up the response - extract just the code if wrapped in markdown code blocks
    code_block_pattern = re.compile(r'```(?:python)?\n(.*?)\n```', re.DOTALL)
    match = code_block_pattern.search(response)
    if match:
        resolved_code = match.group(1)
    else:
        resolved_code = response

    similarity = calculate_similarity(resolved_code, m_content) if resolved_code else 0.0
    print(f"  Similarity to actual merged file: {similarity:.2%}")

    return {
        'conflict_dir': conflict_name,
        'similarity': similarity,
        'inference_time': inference_time,
        'resolved_code': resolved_code
    }

def evaluate_model_performance(conflict_dirs, model_name, prompt, limit=10, use_trained_model=False, peft_model_id=None, output_suffix=""):
    """Evaluate model performance on a subset of conflicts."""
    print(f"\n{'='*50}")
    print(f"EVALUATING {'TRAINED' if use_trained_model else 'BASE'} MODEL")
    print(f"{'='*50}")
    
    # Limit conflicts for evaluation
    eval_conflicts = conflict_dirs[:limit] if limit > 0 else conflict_dirs
    print(f"Evaluating on {len(eval_conflicts)} conflicts")
    
    results = []
    total_similarity = 0.0
    total_inference_time = 0.0
    successful_resolutions = 0
    
    # Create output directory for resolutions
    resolutions_dir = os.path.join("resolution" + output_suffix)
    os.makedirs(resolutions_dir, exist_ok=True)

    for i, conflict_dir in enumerate(eval_conflicts, 1):
        print(f"\nProcessing conflict {i}/{len(eval_conflicts)}: {os.path.basename(conflict_dir)}")
        result = analyze_conflict(conflict_dir, model_name, prompt, use_trained_model, peft_model_id)
        
        if result:
            results.append(result)

            # Track statistics
            if result['similarity'] > 0:
                total_similarity += result['similarity']
                successful_resolutions += 1

            total_inference_time += result['inference_time']

            # Save resolution files
            conflict_name = os.path.basename(conflict_dir)
            conflict_resolution_dir = os.path.join(resolutions_dir, conflict_name)
            os.makedirs(conflict_resolution_dir, exist_ok=True)

            try:
                # Save model's resolved code
                resolved_file_path = os.path.join(conflict_resolution_dir, f"resolved{output_suffix}.py")
                with open(resolved_file_path, 'w', encoding='utf-8') as f:
                    f.write(result['resolved_code'])
                print(f"  Saved model resolution to {resolved_file_path}")

                # Save original files for reference (only on first run)
                if not use_trained_model:
                    merged_path = os.path.join(conflict_dir, 'merged.py')
                    m_path = os.path.join(conflict_dir, 'M.py')
                    
                    merged_content = read_file_content(merged_path)
                    m_content = read_file_content(m_path)
                    
                    # Save original merged file with conflicts
                    merged_file_path = os.path.join(conflict_resolution_dir, "merged.py")
                    with open(merged_file_path, 'w', encoding='utf-8') as f:
                        f.write(merged_content)
                    
                    # Save correct resolution
                    correct_file_path = os.path.join(conflict_resolution_dir, "M.py")
                    with open(correct_file_path, 'w', encoding='utf-8') as f:
                        f.write(m_content)

            except Exception as e:
                print(f"  Error saving files: {e}")

    # Calculate average metrics
    avg_similarity = total_similarity / successful_resolutions if successful_resolutions > 0 else 0.0
    avg_inference_time = total_inference_time / len(results) if results else 0.0

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS - {'TRAINED' if use_trained_model else 'BASE'} MODEL")
    print(f"{'='*50}")
    print(f"Total conflicts processed: {len(results)}")
    print(f"Successful resolutions: {successful_resolutions}")
    print(f"Average similarity to ground truth: {avg_similarity:.2%}")
    print(f"Average inference time: {avg_inference_time:.2f} seconds")

    # Save results to JSON
    output_data = {
        'model_type': 'trained' if use_trained_model else 'base',
        'summary': {
            'total_conflicts': len(results),
            'successful_resolutions': successful_resolutions,
            'average_similarity': avg_similarity,
            'average_inference_time': avg_inference_time,
        },
        'results': results
    }

    output_file = f"evaluation_results{'_trained' if use_trained_model else '_base'}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_file}")
    
    return avg_similarity, avg_inference_time

def prepare_training_data(conflict_dirs):
    """Prepare training data from conflict directories."""
    training_data = []
    
    system_prompt = """You are an expert developer tasked with resolving merge conflicts.

I'll provide you with a file that contains Git merge conflicts marked with the standard conflict markers.

Your task is to:
1. Analyze the conflict markers
2. Resolve each conflict by determining the best way to merge the changes
3. Output the complete file with all conflicts resolved
4. Remove all conflict markers in your output

Please provide ONLY the complete resolved code file with no explanations or additional text."""
    
    for conflict_dir in conflict_dirs:
        conflict_name = os.path.basename(conflict_dir)
        
        # Get file paths
        merged_path = os.path.join(conflict_dir, 'merged.py')
        m_path = os.path.join(conflict_dir, 'M.py')
        
        # Read file contents
        merged_content = read_file_content(merged_path)
        m_content = read_file_content(m_path)
        
        if not merged_content or not m_content:
            print(f"  Error: Could not read required files for {conflict_name}")
            continue
        
        # Add to training data
        training_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": merged_content},
                {"role": "assistant", "content": m_content}
            ]
        })
    
    return training_data

def main():
    parser = argparse.ArgumentParser(description='Fine-tune model for merge conflict resolution using LoRA')
    parser.add_argument('--dataset', type=str, default='training_data_py', help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='lora_merge_resolver', help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--max_samples', type=int, default=0, help='Maximum number of samples to use (0 for all)')
    parser.add_argument('--disable_quantization', action='store_true', help='Disable 4-bit quantization (for Mac/CPU compatibility)')
    parser.add_argument('--eval_limit', type=int, default=10, help='Number of conflicts to evaluate on (0 for all)')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only evaluate existing model')
    args = parser.parse_args()
    
    # Get conflict directories
    dataset_path = os.path.abspath(args.dataset)
    print(f"Looking for conflicts in {dataset_path}")
    
    conflict_dirs = get_conflict_dirs(dataset_path)
    if not conflict_dirs:
        print(f"No conflict directories found in {dataset_path}")
        sys.exit(1)
    
    print(f"Found {len(conflict_dirs)} conflict directories")
    
    # Limit the number of samples if specified
    if args.max_samples > 0 and args.max_samples < len(conflict_dirs):
        conflict_dirs = conflict_dirs[:args.max_samples]
        print(f"Limiting to {args.max_samples} samples")
    
    # Define evaluation prompt
    eval_prompt = """Resolve the Git merge conflicts in the provided file. Output only the complete, resolved file content with all conflict markers removed. No explanations or extra text."""
    
    # Evaluate base model performance
    print("\n" + "="*60)
    print("BASELINE EVALUATION - Testing base model performance")
    print("="*60)
    base_similarity, base_time = evaluate_model_performance(
        conflict_dirs, MODEL_NAME, eval_prompt, 
        limit=args.eval_limit, use_trained_model=False, 
        output_suffix="_base"
    )
    
    if not args.skip_training:
        # Prepare training data
        print("\n" + "="*60)
        print("TRAINING PHASE")
        print("="*60)
        print("Preparing training data...")
        training_data = prepare_training_data(conflict_dirs)
        print(f"Prepared {len(training_data)} training samples")
        
        # Create dataset
        dataset = Dataset.from_list(training_data)
        
        # Load tokenizer
        print(f"Loading tokenizer for {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization based on hardware compatibility
        if args.disable_quantization or not torch.cuda.is_available():
            print("Running without quantization (using float16 or CPU)")
            bnb_config = None
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            # Configure quantization for CUDA GPUs
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            device_map = "auto"
            dtype = None
        
        # Load model with appropriate configuration
        print(f"Loading model {MODEL_NAME}...")
        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": True,
        }
        
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        if dtype:
            model_kwargs["torch_dtype"] = dtype
            
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            **model_kwargs
        )
        
        # Only prepare for kbit training if using quantization
        if bnb_config:
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Get PEFT model
        model = get_peft_model(model, peft_config)
        
        # Adjust training arguments based on hardware
        fp16 = torch.cuda.is_available() and not args.disable_quantization
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            fp16=fp16,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="no",
            save_total_limit=3,
            load_best_model_at_end=False,
            report_to="none"
        )
        
        # Create SFT trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            peft_config=peft_config
        )
        
        # Train model
        print("Starting training...")
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        # Save model
        print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")
        print(f"Saving model to {args.output_dir}")
        trainer.save_model()
        
        print("Training completed successfully!")
    
    # Evaluate trained model performance
    print("\n" + "="*60)
    print("POST-TRAINING EVALUATION - Testing trained model performance")
    print("="*60)
    trained_similarity, trained_time = evaluate_model_performance(
        conflict_dirs, MODEL_NAME, eval_prompt,
        limit=args.eval_limit, use_trained_model=True, 
        peft_model_id=args.output_dir, output_suffix="_trained"
    )
    
    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"Base Model Similarity:    {base_similarity:.2%}")
    print(f"Trained Model Similarity: {trained_similarity:.2%}")
    print(f"Improvement:              {(trained_similarity - base_similarity):.2%}")
    print(f"Base Model Avg Time:      {base_time:.2f}s")
    print(f"Trained Model Avg Time:   {trained_time:.2f}s")
    
    if trained_similarity > base_similarity:
        print("✅ Training improved model performance!")
    else:
        print("❌ Training did not improve model performance.")

if __name__ == "__main__":
    main() 