#!/usr/bin/env python3
import os
import sys
import json
import argparse
import difflib
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("merge_resolver.log")
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
MAX_TOKENS = 32768

class MergeConflictResolver:
    def __init__(self, model_name: str = MODEL_NAME, max_tokens: int = MAX_TOKENS):
        """
        Initialize the merge conflict resolver with the specified model.
        
        Args:
            model_name: The name of the model to use for conflict resolution
            max_tokens: Maximum number of tokens for model generation
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None
        
    async def initialize_model(self):
        """Initialize the model and tokenizer asynchronously."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            config = AutoConfig.from_pretrained(self.model_name)
            
            # Load model in a separate thread to avoid blocking
            self.model = await asyncio.to_thread(
                AutoModelForCausalLM.from_pretrained,
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                config=config
            )
            
            self.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                self.model_name
            )
            
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            return False
    
    async def resolve_conflict(self, conflict_content: str) -> Tuple[str, float]:
        """
        Resolve a merge conflict using the loaded model.
        
        Args:
            conflict_content: Content of the file with merge conflicts
            
        Returns:
            Tuple containing the resolved code and inference time
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        prompt = """
You are an expert developer tasked with resolving merge conflicts.

I'll provide you with a file that contains Git merge conflicts marked with the standard conflict markers.

Your task is to:
1. Analyze the conflict markers
2. Resolve each conflict by determining the best way to merge the changes
3. Output the complete file with all conflicts resolved
4. Remove all conflict markers in your output

Please provide ONLY the complete resolved code file with no explanations or additional text.
"""
        
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": conflict_content}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            start_time = time.time()
            
            generated_ids = await asyncio.to_thread(
                self.model.generate,
                **model_inputs,
                max_new_tokens=self.max_tokens
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            return response, inference_time
        except Exception as e:
            logger.error(f"Error during conflict resolution: {str(e)}")
            return "", 0.0

class ConflictDataset:
    def __init__(self, base_dir: str):
        """
        Initialize the conflict dataset.
        
        Args:
            base_dir: Path to the directory containing conflict directories
        """
        self.base_dir = Path(base_dir)
        
    def get_conflict_dirs(self) -> List[Path]:
        """
        Get all conflict directories in the dataset.
        
        Returns:
            List of paths to conflict directories
        """
        conflict_dirs = []
        
        try:
            # Check if base_dir exists
            if not self.base_dir.exists():
                logger.error(f"Error: Directory {self.base_dir} does not exist.")
                return []
            
            # List all subdirectories in the dataset directory
            for conflict_dir in self.base_dir.iterdir():
                # Skip non-directories and hidden files
                if not conflict_dir.is_dir() or conflict_dir.name.startswith('.'):
                    continue
                    
                # Check if the directory contains merged.py and M.py
                if (conflict_dir / 'merged.py').exists() and (conflict_dir / 'M.py').exists():
                    conflict_dirs.append(conflict_dir)
            
            return conflict_dirs
        except Exception as e:
            logger.error(f"Error getting conflict directories: {str(e)}")
            return []
    
    @staticmethod
    def read_file_content(file_path: Path) -> Optional[str]:
        """
        Read the content of a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Content of the file or None if an error occurred
        """
        try:
            return file_path.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None

class ConflictAnalyzer:
    @staticmethod
    def calculate_similarity(file1_content: str, file2_content: str) -> float:
        """
        Calculate similarity between two files using difflib.
        
        Args:
            file1_content: Content of the first file
            file2_content: Content of the second file
            
        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        if not file1_content or not file2_content:
            return 0.0
            
        matcher = difflib.SequenceMatcher(None, file1_content, file2_content)
        return matcher.ratio()
    
    @staticmethod
    def get_diff(file1_content: str, file2_content: str, file1_name: str, file2_name: str) -> Optional[str]:
        """
        Get the diff between two files.
        
        Args:
            file1_content: Content of the first file
            file2_content: Content of the second file
            file1_name: Name of the first file
            file2_name: Name of the second file
            
        Returns:
            Unified diff between the files or None if an error occurred
        """
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

class ResultsManager:
    def __init__(self, output_file: str, resolutions_dir: str):
        """
        Initialize the results manager.
        
        Args:
            output_file: Path to the output JSON file
            resolutions_dir: Directory to save resolution files
        """
        self.output_file = output_file
        self.resolutions_dir = Path(resolutions_dir)
        self.results = []
        self.summary = {
            'total_conflicts': 0,
            'successful_resolutions': 0,
            'average_similarity': 0.0,
            'average_inference_time': 0.0,
            'total_inference_time': 0.0
        }
        
        # Create output directory
        self.resolutions_dir.mkdir(exist_ok=True)
    
    def add_result(self, result: Dict[str, Any]):
        """
        Add a result to the results list.
        
        Args:
            result: Result dictionary
        """
        self.results.append(result)
        
        # Update summary statistics
        self.summary['total_conflicts'] += 1
        if result['similarity'] > 0:
            self.summary['successful_resolutions'] += 1
        self.summary['total_inference_time'] += result['inference_time']
    
    def save_resolution_files(self, conflict_name: str, merged_content: str, 
                             m_content: str, resolved_code: str):
        """
        Save resolution files for a conflict.
        
        Args:
            conflict_name: Name of the conflict directory
            merged_content: Content of the merged.py file
            m_content: Content of the M.py file
            resolved_code: Resolved code from the model
        """
        try:
            # Create a directory for this specific conflict
            conflict_resolution_dir = self.resolutions_dir / conflict_name
            conflict_resolution_dir.mkdir(exist_ok=True)
            
            # Save model's resolved code
            resolved_file_path = conflict_resolution_dir / "resolved.py"
            resolved_file_path.write_text(resolved_code, encoding='utf-8')
            
            # Save original merged file with conflicts
            merged_file_path = conflict_resolution_dir / "merged.py"
            merged_file_path.write_text(merged_content, encoding='utf-8')
            
            # Save correct resolution
            correct_file_path = conflict_resolution_dir / "M.py"
            correct_file_path.write_text(m_content, encoding='utf-8')
            
            logger.info(f"Saved resolution files to {conflict_resolution_dir}")
        except Exception as e:
            logger.error(f"Error saving resolution files: {str(e)}")
    
    def finalize(self):
        """
        Finalize the results and save to the output file.
        """
        # Calculate average metrics
        if self.summary['successful_resolutions'] > 0:
            self.summary['average_similarity'] = sum(r['similarity'] for r in self.results) / self.summary['successful_resolutions']
        
        if self.summary['total_conflicts'] > 0:
            self.summary['average_inference_time'] = self.summary['total_inference_time'] / self.summary['total_conflicts']
        
        # Save results
        output_data = {
            'summary': self.summary,
            'results': self.results
        }
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Results saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

async def analyze_conflict(resolver: MergeConflictResolver, conflict_dir: Path, 
                          results_manager: ResultsManager):
    """
    Analyze a conflict directory using the model.
    
    Args:
        resolver: MergeConflictResolver instance
        conflict_dir: Path to the conflict directory
        results_manager: ResultsManager instance
    """
    conflict_name = conflict_dir.name
    logger.info(f"Analyzing conflict: {conflict_name}")
    
    # Check for the expected files
    merged_path = conflict_dir / 'merged.py'
    m_path = conflict_dir / 'M.py'
    
    # Read file contents
    merged_content = ConflictDataset.read_file_content(merged_path)
    m_content = ConflictDataset.read_file_content(m_path)
    
    if not merged_content or not m_content:
        logger.error(f"Could not read required files for {conflict_name}")
        result = {
            'conflict_dir': conflict_name,
            'similarity': 0,
            'inference_time': 0,
            'resolved_code': "None"
        }
        results_manager.add_result(result)
        return
    
    # Resolve the conflict
    resolved_code, inference_time = await resolver.resolve_conflict(merged_content)
    logger.info(f"Resolution completed in {inference_time:.2f} seconds")
    
    # Calculate similarity with the actual merged file
    similarity = ConflictAnalyzer.calculate_similarity(resolved_code, m_content) if resolved_code else 0.0
    logger.info(f"Similarity to actual merged file: {similarity:.2%}")
    
    # Create result
    result = {
        'conflict_dir': conflict_name,
        'similarity': similarity,
        'inference_time': inference_time,
        'resolved_code': resolved_code
    }
    
    # Add result and save files
    results_manager.add_result(result)
    results_manager.save_resolution_files(conflict_name, merged_content, m_content, resolved_code)

async def main():
    """Main function to run the merge conflict resolution analysis."""
    parser = argparse.ArgumentParser(description='Resolve merge conflicts using AI models')
    parser.add_argument('--dataset', type=str, default='training_data_py', help='Path to the dataset directory')
    parser.add_argument('--limit', type=int, default=10, help='Limit the number of conflicts to analyze (0 for all)')
    parser.add_argument('--output', type=str, default='resolution_results.json', help='Output file for analysis results')
    parser.add_argument('--model', type=str, default=MODEL_NAME, help='Model to use for conflict resolution')
    parser.add_argument('--resolutions-dir', type=str, default='resolutions', help='Directory to save resolution files')
    args = parser.parse_args()
    
    try:
        # Initialize dataset
        dataset = ConflictDataset(args.dataset)
        conflict_dirs = dataset.get_conflict_dirs()
        
        if not conflict_dirs:
            logger.error(f"No conflict directories found in {args.dataset}")
            return 1
        
        logger.info(f"Found {len(conflict_dirs)} conflict directories")
        
        # Limit the number of conflicts to analyze
        if args.limit > 0:
            conflict_dirs = conflict_dirs[:args.limit]
            logger.info(f"Limiting analysis to {args.limit} conflicts")
        
        # Initialize results manager
        results_manager = ResultsManager(args.output, args.resolutions_dir)
        
        # Initialize model
        resolver = MergeConflictResolver(model_name=args.model)
        if not await resolver.initialize_model():
            logger.error("Failed to initialize model. Exiting.")
            return 1
        
        # Process conflicts
        for i, conflict_dir in enumerate(conflict_dirs, 1):
            logger.info(f"Processing conflict {i}/{len(conflict_dirs)}: {conflict_dir.name}")
            await analyze_conflict(resolver, conflict_dir, results_manager)
        
        # Finalize results
        results_manager.finalize()
        
        # Print summary
        logger.info(f"\nAnalysis completed:")
        logger.info(f"Total conflicts: {results_manager.summary['total_conflicts']}")
        logger.info(f"Successfully resolved: {results_manager.summary['successful_resolutions']}")
        logger.info(f"Average similarity: {results_manager.summary['average_similarity']:.2%}")
        logger.info(f"Average inference time: {results_manager.summary['average_inference_time']:.2f} seconds")
        
        return 0
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 