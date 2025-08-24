#!/usr/bin/env python3
import os
import sys
import json
import argparse
import difflib
from pathlib import Path
import requests
import time
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline, AutoConfig
import torch
from peft import PeftModel


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

def get_diff(file1_content, file2_content, file1_name, file2_name):
    """Get the diff between two files."""
    if file1_content is None or file2_content is None:
        return None
        
    file1_lines = file1_content.splitlines()
    file2_lines = file2_content.splitlines()
    
    diff = difflib.unified_diff(
        file1_lines, 
        file2_lines,
        fromfile=file1_name,
        tofile=file2_name,
        lineterm=''
    )
    
    return '\n'.join(diff)

def calculate_similarity(file1_content, file2_content):
    """Calculate similarity between two files using difflib."""
    if not file1_content or not file2_content:
        return 0.0
        
    matcher = difflib.SequenceMatcher(None, file1_content, file2_content)
    return matcher.ratio()

def analyze_conflict(conflict_dir, model_name=MODEL_NAME, use_trained_model=False, peft_model_id=None, custom_prompt=None):
    """Analyze a conflict directory using the specified model."""
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
            'resolved_code': "None",
            'model_type': 'trained' if use_trained_model else 'base'
        }
    
    # Default prompt if none provided
    if custom_prompt is None:
        prompt = """You are an expert developer tasked with resolving merge conflicts.

I'll provide you with a file that contains Git merge conflicts marked with the standard conflict markers.

Your task is to:
1. Analyze the conflict markers
2. Resolve each conflict by determining the best way to merge the changes
3. Output the complete file with all conflicts resolved
4. Remove all conflict markers in your output

Please provide ONLY the complete resolved code file with no explanations or additional text."""
    else:
        prompt = custom_prompt
    

    # Load model with error handling
    try:
        print(f"  Loading {'trained' if use_trained_model else 'base'} model...")
        config = AutoConfig.from_pretrained(model_name)
        
        # Load base model
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto"
        }
        
        # Only add flash attention if available
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
                attn_implementation="flash_attention_2"
            )
        except Exception:
            # Fallback without flash attention
            print("  Flash attention not available, using default attention")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        
        # Load fine-tuned model if specified
        if use_trained_model and peft_model_id:
            if not os.path.exists(peft_model_id):
                print(f"  Error: Trained model path {peft_model_id} does not exist")
                return {
                    'conflict_dir': conflict_name,
                    'similarity': 0,
                    'inference_time': 0,
                    'resolved_code': "Model not found",
                    'model_type': 'trained',
                    'error': f"Model path {peft_model_id} not found"
                }
            model = PeftModel.from_pretrained(model, peft_model_id)
            print(f"  Loaded trained model from {peft_model_id}")
        
    except Exception as e:
        print(f"  Error loading model: {e}")
        return {
            'conflict_dir': conflict_name,
            'similarity': 0,
            'inference_time': 0,
            'resolved_code': "Model loading failed",
            'model_type': 'trained' if use_trained_model else 'base',
            'error': str(e)
        }

    # Prepare messages and tokenizer
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

    try:
        generated_ids = model.generate(
          **model_inputs,
           max_new_tokens=MAX_TOKENS,
           do_sample=False,  # Deterministic output
           pad_token_id=tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"  Error during model inference: {e}")
        return {
            'conflict_dir': conflict_name,
            'similarity': 0,
            'inference_time': 0,
            'resolved_code': "Inference failed",
            'model_type': 'trained' if use_trained_model else 'base',
            'error': str(e)
        }

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"  Resolution completed in {inference_time:.2f} seconds")
    
    # Clean up the response - extract just the code if wrapped in markdown code blocks
    code_block_pattern = re.compile(r'```(?:python)?\n(.*?)\n```', re.DOTALL)
    match = code_block_pattern.search(response)
    if match:
        resolved_code = match.group(1)
        print("  Extracted code from markdown blocks")
    else:
        resolved_code = response.strip()
    
    # Calculate similarity with the actual merged file
    similarity = calculate_similarity(resolved_code, m_content) if resolved_code else 0.0
    print(f"  Similarity to ground truth: {similarity:.2%}")
    
    return {
        'conflict_dir': conflict_name,
        'similarity': similarity,
        'inference_time': inference_time,
        'resolved_code': resolved_code,
        'model_type': 'trained' if use_trained_model else 'base'
    }

def compare_models(conflict_dirs, base_results, trained_results):
    """Compare performance between base and trained models."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON ANALYSIS")
    print(f"{'='*60}")
    
    improvements = 0
    degradations = 0
    no_change = 0
    
    detailed_comparison = []
    
    for base_result, trained_result in zip(base_results, trained_results):
        if base_result['conflict_dir'] == trained_result['conflict_dir']:
            base_sim = base_result['similarity']
            trained_sim = trained_result['similarity']
            improvement = trained_sim - base_sim
            
            status = "IMPROVED" if improvement > 0.01 else "DEGRADED" if improvement < -0.01 else "NO_CHANGE"
            
            detailed_comparison.append({
                'conflict': base_result['conflict_dir'],
                'base_similarity': base_sim,
                'trained_similarity': trained_sim,
                'improvement': improvement,
                'status': status
            })
            
            if improvement > 0.01:
                improvements += 1
            elif improvement < -0.01:
                degradations += 1
            else:
                no_change += 1
    
    print(f"Conflicts with improvement: {improvements}")
    print(f"Conflicts with degradation: {degradations}")
    print(f"Conflicts with no change: {no_change}")
    print(f"Total conflicts compared: {len(detailed_comparison)}")
    
    # Show top improvements and degradations
    detailed_comparison.sort(key=lambda x: x['improvement'], reverse=True)
    
    print(f"\nTop 5 Improvements:")
    for i, comp in enumerate(detailed_comparison[:5]):
        print(f"  {i+1}. {comp['conflict']}: {comp['base_similarity']:.2%} → {comp['trained_similarity']:.2%} (+{comp['improvement']:.2%})")
    
    print(f"\nTop 5 Degradations:")
    for i, comp in enumerate(detailed_comparison[-5:]):
        print(f"  {i+1}. {comp['conflict']}: {comp['base_similarity']:.2%} → {comp['trained_similarity']:.2%} ({comp['improvement']:.2%})")
    
    return detailed_comparison

def save_results(results, output_file, model_type="base"):
    """Save results to JSON file."""
    # Calculate metrics
    total_similarity = sum(r['similarity'] for r in results if 'error' not in r)
    total_inference_time = sum(r['inference_time'] for r in results if 'error' not in r)
    successful_resolutions = len([r for r in results if r['similarity'] > 0 and 'error' not in r])
    failed_resolutions = len([r for r in results if 'error' in r])
    
    avg_similarity = total_similarity / successful_resolutions if successful_resolutions > 0 else 0.0
    avg_inference_time = total_inference_time / len(results) if results else 0.0
    
    summary = {
        'model_type': model_type,
        'total_conflicts': len(results),
        'successful_resolutions': successful_resolutions,
        'failed_resolutions': failed_resolutions,
        'average_similarity': avg_similarity,
        'average_inference_time': avg_inference_time,
        'total_inference_time': total_inference_time
    }
    
    output_data = {
        'summary': summary,
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return summary

def main():
    parser = argparse.ArgumentParser(description='Test merge conflict resolution models')
    parser.add_argument('--dataset', type=str, default='training_data_py', help='Path to the dataset directory')
    parser.add_argument('--limit', type=int, default=10, help='Limit the number of conflicts to analyze (0 for all)')
    parser.add_argument('--output', type=str, default='testing_results.json', help='Output file for analysis results')
    parser.add_argument('--save-files', action='store_true', help='Save model resolutions to separate files')
    parser.add_argument('--model-name', type=str, default=MODEL_NAME, help='Base model name to use')
    parser.add_argument('--trained-model', type=str, help='Path to trained model (PEFT adapter)')
    parser.add_argument('--compare', action='store_true', help='Compare base model vs trained model')
    parser.add_argument('--base-only', action='store_true', help='Test only base model')
    parser.add_argument('--trained-only', action='store_true', help='Test only trained model')
    parser.add_argument('--custom-prompt', type=str, help='Custom prompt to use for testing')
    parser.add_argument('--output-dir', type=str, default='test_resolutions', help='Directory to save resolution files')
    args = parser.parse_args()
    
    # Validate arguments
    if args.trained_only and not args.trained_model:
        print("Error: --trained-only requires --trained-model to be specified")
        sys.exit(1)
    
    if args.compare and not args.trained_model:
        print("Error: --compare requires --trained-model to be specified")
        sys.exit(1)
    
    # Get conflict directories
    dataset_path = os.path.abspath(args.dataset)
    print(f"Looking for conflicts in {dataset_path}")
    
    conflict_dirs = get_conflict_dirs(dataset_path)
    if not conflict_dirs:
        print(f"No conflict directories found in {dataset_path}")
        sys.exit(1)
    
    print(f"Found {len(conflict_dirs)} conflict directories")
    
    # Limit the number of conflicts to analyze
    if args.limit > 0:
        conflict_dirs = conflict_dirs[:args.limit]
        print(f"Limiting analysis to {args.limit} conflicts")
    
    # Create output directory for resolutions if needed
    if args.save_files:
        os.makedirs(args.output_dir, exist_ok=True)
    
    base_results = []
    trained_results = []
    
    # Test base model (unless trained-only)
    if not args.trained_only:
        print(f"\n{'='*60}")
        print("TESTING BASE MODEL")
        print(f"{'='*60}")
        
        for i, conflict_dir in enumerate(conflict_dirs, 1):
            print(f"\nProcessing conflict {i}/{len(conflict_dirs)}: {os.path.basename(conflict_dir)}")
            result = analyze_conflict(
                conflict_dir, 
                model_name=args.model_name, 
                use_trained_model=False,
                custom_prompt=args.custom_prompt
            )
            if result:
                base_results.append(result)
                
                # Save files if requested
                if args.save_files:
                    conflict_name = os.path.basename(conflict_dir)
                    conflict_resolution_dir = os.path.join(args.output_dir, conflict_name)
                    os.makedirs(conflict_resolution_dir, exist_ok=True)
                    
                    try:
                        # Save model's resolved code
                        resolved_file_path = os.path.join(conflict_resolution_dir, "resolved_base.py")
                        with open(resolved_file_path, 'w', encoding='utf-8') as f:
                            f.write(result['resolved_code'])
                        print(f"  Saved base model resolution to {resolved_file_path}")
                        
                        # Save reference files (only once)
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
        
        # Save base model results
        base_output = args.output.replace('.json', '_base.json')
        base_summary = save_results(base_results, base_output, "base")
        
        print(f"\nBASE MODEL SUMMARY:")
        print(f"Average similarity: {base_summary['average_similarity']:.2%}")
        print(f"Average inference time: {base_summary['average_inference_time']:.2f} seconds")
        print(f"Success rate: {base_summary['successful_resolutions']}/{base_summary['total_conflicts']}")
    
    # Test trained model (if specified and not base-only)
    if args.trained_model and not args.base_only:
        print(f"\n{'='*60}")
        print("TESTING TRAINED MODEL")
        print(f"{'='*60}")
        
        for i, conflict_dir in enumerate(conflict_dirs, 1):
            print(f"\nProcessing conflict {i}/{len(conflict_dirs)}: {os.path.basename(conflict_dir)}")
            result = analyze_conflict(
                conflict_dir,
                model_name=args.model_name,
                use_trained_model=True,
                peft_model_id=args.trained_model,
                custom_prompt=args.custom_prompt
            )
            if result:
                trained_results.append(result)
                
                # Save files if requested
                if args.save_files:
                    conflict_name = os.path.basename(conflict_dir)
                    conflict_resolution_dir = os.path.join(args.output_dir, conflict_name)
                    os.makedirs(conflict_resolution_dir, exist_ok=True)
                    
                    try:
                        # Save model's resolved code
                        resolved_file_path = os.path.join(conflict_resolution_dir, "resolved_trained.py")
                        with open(resolved_file_path, 'w', encoding='utf-8') as f:
                            f.write(result['resolved_code'])
                        print(f"  Saved trained model resolution to {resolved_file_path}")
                        
                    except Exception as e:
                        print(f"  Error saving files: {e}")
        
        # Save trained model results
        trained_output = args.output.replace('.json', '_trained.json')
        trained_summary = save_results(trained_results, trained_output, "trained")
        
        print(f"\nTRAINED MODEL SUMMARY:")
        print(f"Average similarity: {trained_summary['average_similarity']:.2%}")
        print(f"Average inference time: {trained_summary['average_inference_time']:.2f} seconds")
        print(f"Success rate: {trained_summary['successful_resolutions']}/{trained_summary['total_conflicts']}")
    
    # Compare models if both were tested
    if args.compare and base_results and trained_results:
        detailed_comparison = compare_models(conflict_dirs, base_results, trained_results)
        
        # Save comparison results
        comparison_output = args.output.replace('.json', '_comparison.json')
        comparison_data = {
            'base_summary': base_summary,
            'trained_summary': trained_summary,
            'detailed_comparison': detailed_comparison,
            'overall_improvement': trained_summary['average_similarity'] - base_summary['average_similarity']
        }
        
        with open(comparison_output, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\nOVERALL COMPARISON:")
        print(f"Base model avg similarity:    {base_summary['average_similarity']:.2%}")
        print(f"Trained model avg similarity: {trained_summary['average_similarity']:.2%}")
        improvement = trained_summary['average_similarity'] - base_summary['average_similarity']
        print(f"Overall improvement:          {improvement:+.2%}")
        
        if improvement > 0:
            print("✅ Training improved model performance!")
        else:
            print("❌ Training did not improve model performance.")
        
        print(f"Comparison results saved to {comparison_output}")
    
    print(f"\nTesting completed!")

if __name__ == "__main__":
    main()