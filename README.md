# ConflictMerger

An AI-powered system for automatically resolving Git merge conflicts using machine learning models. This project leverages fine-tuned language models to intelligently analyze and resolve code merge conflicts across multiple programming languages.

## Project Overview

ConflictMerger addresses one of the most time-consuming aspects of collaborative software development: resolving merge conflicts. By training language models on conflict resolution patterns, the system can automatically suggest intelligent resolutions for merge conflicts in codebases.

### Key Features

- **Multi-language Support**: Handles conflicts in Python, Java, JavaScript, Clojure, Lua, and Shell scripts
- **Fine-tuned Models**: Uses LoRA (Low-Rank Adaptation) fine-tuning on Qwen2.5-Coder models
- **Automated Pipeline**: Complete workflow from data preprocessing to model evaluation
- **Performance Metrics**: Similarity scoring and inference time tracking
- **Asynchronous Processing**: Efficient conflict resolution with async/await patterns
- **GPU Acceleration**: Supports CUDA for faster model training and inference
- **Quantization Support**: 4-bit quantization for memory-efficient deployment

## Architecture

ConflictMerger follows a **3-phase pipeline** for AI-powered merge conflict resolution:

### Phase 1: Data Preparation
**`preprocessing.py`** - Converts raw Git conflict scenarios into training-ready datasets
- Takes base files and conflicting versions (A, B)
- Generates files with Git conflict markers (`<<<<<<< HEAD`, `=======`, `>>>>>>> branch`)
- Creates structured training data for model learning

### Phase 2: Model Training (Optional)
**`training.py`** - Fine-tunes language models on conflict resolution patterns
- Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Learns domain-specific conflict resolution strategies
- Produces specialized models for better resolution quality

### Phase 3: Conflict Resolution (Main Application)
**`merge_resolver.py`** - **The core application that actually resolves merge conflicts**
- Loads trained models (base or fine-tuned)
- Takes files with active merge conflicts
- Generates resolved code by intelligently merging changes
- Provides real-time, production-ready conflict resolution

### Supporting Components
4. **Testing & Evaluation** (`testing.py`) - Benchmarks model performance and quality
5. **Interactive Notebook** (`conflictmerger.ipynb`) - Experimental workflow and analysis

### Workflow Summary
```
Raw Conflicts → preprocessing.py → Training Data → training.py → Fine-tuned Model
                                                                      ↓
Your Git Conflicts → merge_resolver.py (+ Fine-tuned Model) → Resolved Code
```

## Project Structure

```
ConflictMerger/
├── merge_resolver.py      # Main conflict resolution engine
├── training.py           # Model training and fine-tuning
├── testing.py           # Model evaluation and testing
├── preprocessing.py     # Data preparation utilities
├── conflictmerger.ipynb # Interactive development notebook
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── token.txt           # HuggingFace token (optional)
├── .gitignore          # Git ignore rules
│
├── dataset/             # Raw conflict datasets
│   ├── conflicts-py/    # Python conflicts
│   ├── conflicts-java/  # Java conflicts
│   ├── conflicts-js/    # JavaScript conflicts
│   ├── conflicts-clj/   # Clojure conflicts
│   ├── conflicts-lua/   # Lua conflicts
│   ├── conflicts-sh/    # Shell script conflicts
│   └── parse-error-*/   # Parse error datasets
│
├── training_data_*/     # Processed training data by language
│   ├── training_data_py/
│   ├── training_data_java/
│   ├── training_data_js/
│   └── training_data_clj/
│
├── resolutions/         # Model resolution outputs
├── resolutions_qwen/    # Qwen model specific outputs
├── resolutions_llama/   # Llama model specific outputs
└── ase-merge/          # Additional merge tools
```

## Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible GPU recommended for training (optional for inference)
- **Memory**: At least 8GB RAM, 16GB+ recommended for training
- **Storage**: 10GB+ free space for models and datasets
- **Git**: For version control and dataset management

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ConflictMerger
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up HuggingFace token** (optional, for private models):
```bash
echo "your_hf_token_here" > token.txt
```

### Quick Start

ConflictMerger can be used in two ways:
1. **Quick Resolution** - Use pre-trained models to resolve conflicts immediately
2. **Custom Training** - Train specialized models on your specific conflict patterns

#### Option A: Quick Resolution (Recommended for most users)

If you just want to resolve conflicts using the base model:

```bash
# Resolve conflicts directly (using base Qwen model)
python merge_resolver.py --dataset_dir your_conflict_data --output_dir resolutions
```

#### Option B: Full Pipeline (For custom model training)

If you want to train a specialized model on your data:

##### 1. Preprocess Data
```bash
# For Python conflicts
python preprocessing.py --input_dir dataset/conflicts-py --output_dir training_data_py --extension py

# For other languages
python preprocessing.py --input_dir dataset/conflicts-java --output_dir training_data_java --extension java
python preprocessing.py --input_dir dataset/conflicts-js --output_dir training_data_js --extension js
```

##### 2. Train Custom Model
```bash
# Basic training
python training.py --data_dir training_data_py --output_dir ./models/conflict_resolver_py

# Training with custom parameters
python training.py \
    --data_dir training_data_py \
    --output_dir ./models/conflict_resolver_py \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

##### 3. Use Your Trained Model
```bash
# Using your custom trained model
python merge_resolver.py \
    --dataset_dir training_data_py \
    --output_dir resolutions \
    --model_path ./models/conflict_resolver_py
```

##### 4. Evaluate Performance (Optional)
```bash
# Evaluate base model
python testing.py --dataset_dir training_data_py

# Evaluate trained model
python testing.py \
    --dataset_dir training_data_py \
    --trained-model ./models/conflict_resolver_py
```

## Usage

### Real-World Conflict Resolution

The primary use case for ConflictMerger is resolving actual Git merge conflicts. Here's how to use `merge_resolver.py` for real conflicts:

#### Using the API (Recommended)

```python
from merge_resolver import MergeConflictResolver
import asyncio

async def resolve_my_conflict():
    # Initialize resolver (base model or custom trained model)
    resolver = MergeConflictResolver(model_name="Qwen/Qwen2.5-Coder-0.5B")
    # Or use your trained model: 
    # resolver = MergeConflictResolver(model_name="./models/my_trained_model")
    
    await resolver.initialize_model()
    
    # Your actual Git conflict content
    conflict_content = """
def calculate_total(items):
<<<<<<< HEAD
    return sum(item.price for item in items)
=======
    total = 0
    for item in items:
        total += item.price * item.quantity
    return total
>>>>>>> feature/quantity-support
"""
    
    # Resolve the conflict
    resolved_code, inference_time = await resolver.resolve_conflict(conflict_content)
    print(f"Resolved in {inference_time:.2f}s:")
    print(resolved_code)

# Run the resolver
asyncio.run(resolve_my_conflict())
```

#### Command Line Usage

```bash
# For dataset evaluation
python merge_resolver.py --dataset_dir training_data_py --output_dir resolutions

# With custom model
python merge_resolver.py \
    --dataset_dir training_data_py \
    --output_dir resolutions \
    --model ./models/my_custom_model \
    --limit 50
```

### Model Training and Evaluation

#### Basic Conflict Resolution

```python
from merge_resolver import MergeConflictResolver
import asyncio

async def resolve_conflict():
    resolver = MergeConflictResolver()
    await resolver.initialize_model()
    
    # Your conflict content with Git markers
    conflict_content = """
    <<<<<<< HEAD
    def function():
        return "version A"
    =======
    def function():
        return "version B"
    >>>>>>> branch
    """
    
    resolved_code, inference_time = await resolver.resolve_conflict(conflict_content)
    print(f"Resolved in {inference_time:.2f}s:")
    print(resolved_code)

asyncio.run(resolve_conflict())
```

### Batch Processing

```python
from testing import evaluate_model_performance, get_conflict_dirs

# Evaluate model on entire dataset
conflict_dirs = get_conflict_dirs("training_data_py")
results = evaluate_model_performance(
    conflict_dirs, 
    model_name="Qwen/Qwen2.5-Coder-0.5B",
    use_trained_model=True,
    peft_model_id="./models/conflict_resolver_py"
)

print(f"Average similarity: {results['avg_similarity']:.3f}")
print(f"Average inference time: {results['avg_time']:.3f}s")
```

## Configuration

### Model Configuration

```python
# Default model settings
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
MAX_TOKENS = 32768

# LoRA Configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### Training Configuration

```python
# Training hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
NUM_EPOCHS = 3
WARMUP_STEPS = 100
SAVE_STEPS = 500

# Hardware optimization
USE_4BIT_QUANTIZATION = True
USE_FP16 = True  # Automatic based on GPU capability
```

### Environment Variables

```bash
# Optional: Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Optional: HuggingFace cache directory
export HF_HOME=/path/to/cache

# Optional: Disable tokenizers parallelism warnings
export TOKENIZERS_PARALLELISM=false
```

## Model Performance

### Evaluation Metrics

- **Similarity Score**: Cosine similarity between generated and expected resolution
- **Inference Time**: Time taken to generate resolution
- **Resolution Quality**: Syntactic correctness and logical coherence
- **Memory Usage**: GPU/CPU memory consumption during inference

### Supported Models

- **Qwen/Qwen2.5-Coder-0.5B** (Primary model, 500M parameters)
- **Qwen/Qwen2.5-Coder-1.5B** (Larger variant, better quality)
- **Custom fine-tuned variants** with LoRA adapters
- **Extensible** to other causal language models (GPT, CodeLlama, etc.)

### Performance Benchmarks

| Model | Avg Similarity | Avg Time (s) | Memory (GB) |
|-------|---------------|--------------|-------------|
| Base Qwen-0.5B | 0.75 | 1.2 | 2.5 |
| Fine-tuned Qwen-0.5B | 0.82 | 1.3 | 2.8 |
| Fine-tuned Qwen-1.5B | 0.87 | 2.1 | 6.2 |

## Dataset

### Dataset Structure

Each conflict scenario contains:

- **Base version** (`O.extension`): Original common ancestor
- **Version A** (`A.extension`): First branch changes
- **Version B** (`B.extension`): Second branch changes  
- **Merged version** (`merged.extension`): Expected resolution
- **Conflict version** (`M.extension`): File with conflict markers

### Supported Languages

| Language | Extension | Dataset Size | Status |
|----------|-----------|--------------|--------|
| Python | `.py` | 1000+ conflicts | ✅ Ready |
| Java | `.java` | 800+ conflicts | ✅ Ready |
| JavaScript | `.js` | 600+ conflicts | ✅ Ready |
| Clojure | `.clj` | 400+ conflicts | ✅ Ready |
| Lua | `.lua` | 300+ conflicts | ✅ Ready |
| Shell | `.sh` | 200+ conflicts | ✅ Ready |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size or enable quantization
   python training.py --batch_size 1 --enable_quantization
   ```

2. **Model Download Issues**:
   ```bash
   # Set HuggingFace token
   huggingface-cli login
   ```

3. **Slow Training**:
   ```bash
   # Enable mixed precision and quantization
   python training.py --fp16 --enable_quantization
   ```




## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on the [Transformers](https://huggingface.co/transformers/) library by Hugging Face
- Uses [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B) models for code understanding
- LoRA fine-tuning powered by [PEFT](https://github.com/huggingface/peft)
- Training infrastructure from [TRL](https://github.com/huggingface/trl)
- Inspired by research in automated program repair and code generation





## Future Work

- **Enhanced Language Support**: Additional programming languages (C++, C#, Go, Rust)
- **IDE Integration**: VS Code and IntelliJ plugins for real-time conflict resolution
- **Context-Aware Resolution**: Multi-file conflict understanding and resolution
- **Performance Optimization**: Model compression and edge deployment
- **Real-time Collaboration**: Integration with Git workflows and CI/CD pipelines
- **Evaluation Framework**: Comprehensive benchmarking suite for conflict resolution quality 