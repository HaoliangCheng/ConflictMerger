import os
import subprocess
import sys
from tqdm import tqdm
import shutil
import argparse

def ensure_dir_exists(dir_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def execute_command(cmd):
    """Execute shell command and handle errors."""
    try:
        p = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return p.stdout
    except subprocess.CalledProcessError as e:
        return None

def get_triplets(base_dir, output_dir):
    """
    Find all triplets of files (base, a, b) in the given directory.
    Assumes files are organized in a structure where each triplet shares a common prefix.
    """
    """
    Find all triplets of files (base, a, b) in the given directory.
    Recursively scans through all subdirectories.
    Returns a dictionary mapping prefixes to their file triplets.
    """
    dir_triplets = {}
    id = 0
    # Walk through all subdirectories
    for root, dirs, _ in os.walk(base_dir):
        for dir in dirs:
            for root, _, files in os.walk(base_dir+'/'+dir):
                id += 1
                for file in files:
                    # Check if we've processed more than 100 files already
                    try:
                        file_len = len(open(os.path.join(root, file), 'r').readlines())
                        if file_len > 500:
                            id -= 1
                            break
                    except Exception as e:
                        break

                    if "O." in file:
                        # prefix = dir
                        if id not in dir_triplets:
                            dir_triplets[id] = {"base": None, "a": None, "b": None}
                        dir_triplets[id]["base"] = os.path.join(root, file)

                    elif "A." in file:
                        # prefix = dir
                        if id not in dir_triplets:
                            dir_triplets[id] = {"base": None, "a": None, "b": None}
                        dir_triplets[id]["a"] = os.path.join(root, file)
                    elif "B." in file:
                        # prefix = dir
                        if id not in dir_triplets:
                            dir_triplets[id] = {"base": None, "a": None, "b": None}
                        dir_triplets[id]["b"] = os.path.join(root, file)
                    else:
                        try:
                            # Create the same directory structure in output_dir
                            # Get the full path of the file
                            file_path = os.path.join(root, file)

                            # Check if the file is empty
                            # Check if the file is empty or contains no actual code (for .py files)
                            is_empty_or_no_code = False
                            try:
                                # First check: Is the file size zero
                                    # Read the file to check for non-comment, non-whitespace lines
                                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                        has_code = False
                                        for line in f:
                                            stripped_line = line.strip()
                                            # Consider a line as code if it's not empty and not a comment
                                            if stripped_line and not stripped_line.startswith('#'):
                                                has_code = True
                                                break # Found code, no need to check further
                                        # If the loop finished without finding code
                                        if not has_code:
                                            is_empty_or_no_code = True
                            except OSError as e: # Catch potential file system errors (e.g., permission denied)
                                print(f"Error accessing file {file_path}: {e}. Skipping.")
                                is_empty_or_no_code = True # Skip if we can't access/read it
                            except Exception as e: # Catch other potential errors during check
                                print(f"Error checking file {file_path}: {e}. Skipping.")
                                is_empty_or_no_code = True # Skip on other errors

                            # If the file is deemed empty or without code, skip to the next iteration
                            if is_empty_or_no_code:
                                id -= 1
                                break # Skip to the next file in the loop
                            output_path = os.path.join(output_dir, str(id))
                            
                            # Ensure the output directory exists
                            if not os.path.exists(output_path):
                                os.makedirs(output_path, exist_ok=True)
                                
                            # Copy the file to the output directory
                            output_file = os.path.join(output_path, file)
                            shutil.copy2(os.path.join(root, file), output_file)
                        except Exception as e:
                            print(f"Error copying file {file}: {str(e)}")
                
    return dir_triplets
    

def git_merge(base_file, a_file, b_file, output_file):
    """
    Use git merge-file to merge the files and create a file with conflict markers.
    
    Args:
        base_file: Path to the base version
        a_file: Path to the a version
        b_file: Path to the b version
        output_file: Path to save the merged result
        temp_dir: Directory for temporary files
    
    Returns:
        True if merge was successful, False otherwise
    """

    # Execute git merge-file command
    merge_cmd = f"git merge-file -L a -L base -L b {base_file} {a_file} {b_file} --diff3 -p > {output_file}"
    result = execute_command(merge_cmd)
    
    # Check if the merge was successful
    return result is not None

def process_dataset(input_dir, output_dir, file_extension):
    """
    Process all file triplets in the input directory and create merged versions.
    
    Args:
        input_dir: Directory containing base, a, b versions
        output_dir: Directory to save merged files
        temp_dir: Directory for temporary files during merge
    """
    # Ensure output directory exists
    ensure_dir_exists(output_dir)
    
    # Get all triplets of files
    triplets = get_triplets(input_dir, output_dir)
    print(f"Found {len(triplets)} complete triplets to process")
    
    successful_merges = 0
    failed_merges = 0
    
    # Process each triplet
    for prefix, files in tqdm(triplets.items(), desc="Processing files"):
        output_file = os.path.join(output_dir, f"{prefix}/merged.{file_extension}")
        
        # Perform git merge
        if git_merge(files["base"], files["a"], files["b"], output_file):
            successful_merges += 1
        else:
            failed_merges += 1
    
    
    print(f"Preprocessing complete. Successful merges: {successful_merges}, Failed merges: {failed_merges}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess files for merge conflict training")
    parser.add_argument('--file_extension', default='py',
                        help='File extension to use for merged files')
    parser.add_argument('--input_dir', default='dataset/conflicts', 
                        help='Directory containing base, a, b versions')
    parser.add_argument('--output_dir', default='training_data',
                        help='Directory to save merged files')
    
    args = parser.parse_args()
    input_dir = args.input_dir + '-' + args.file_extension
    output_dir = args.output_dir + '_' + args.file_extension
    process_dataset(input_dir, output_dir, args.file_extension) 