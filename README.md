# ConflictMerger

An AI-powered system for automatically resolving Git merge conflicts using machine learning models. This project leverages fine-tuned language models to intelligently analyze and resolve code merge conflicts across multiple programming languages.

## 🎯 Project Overview

ConflictMerger addresses one of the most time-consuming aspects of collaborative software development: resolving merge conflicts. By training language models on conflict resolution patterns, the system can automatically suggest intelligent resolutions for merge conflicts in codebases.

### Key Features

- **Multi-language Support**: Handles conflicts in Python, Java, JavaScript, Clojure, Lua, and Shell scripts
- **Fine-tuned Models**: Uses LoRA (Low-Rank Adaptation) fine-tuning on Qwen2.5-Coder models
- **Automated Pipeline**: Complete workflow from data preprocessing to model evaluation
- **Performance Metrics**: Similarity scoring and inference time tracking
- **Asynchronous Processing**: Efficient conflict resolution with async/await patterns

## 🏗️ Architecture

The system consists of several key components:

1. **Data Preprocessing** (`preprocessing.py`) - Prepares training data from conflict datasets
2. **Model Training** (`training.py`) - Fine-tunes language models using LoRA
3. **Conflict Resolution** (`merge_resolver.py`) - Core resolution engine
4. **Testing & Evaluation** (`testing.py`) - Performance evaluation and benchmarking
5. **Interactive Notebook** (`conflictmerger.ipynb`) - Experimental workflow

## 📁 Project Structure

```
main/
├── merge_resolver.py      # Main conflict resolution engine
├── training.py           # Model training and fine-tuning
├── testing.py           # Model evaluation and testing
├── preprocessing.py     # Data preparation utilities
├── conflictmerger.ipynb # Interactive development notebook
├── requirements.txt     # Python dependencies
├── README.md           # This file
│
├── dataset/             # Raw conflict datasets
│   ├── conflicts-py/    # Python conflicts
│   ├── conflicts-java/  # Java conflicts
│   ├── conflicts-js/    # JavaScript conflicts
│   └── ...
│
├── training_data_*/     # Processed training data
├── resolutions/         # Model resolution outputs
├── resolutions_qwen/    # Qwen model specific outputs
└── resolutions_llama/   # Llama model specific outputs
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

1. **Preprocess Data**:
```bash
python preprocessing.py --input_dir dataset/conflicts-py --output_dir training_data_py --extension py
```

2. **Train Model**:
```bash
python training.py --data_dir training_data_py --output_dir ./fine_tuned_model
```

3. **Resolve Conflicts**:
```bash
python merge_resolver.py --dataset_dir training_data_py --output_dir resolutions
```

4. **Evaluate Performance**:
```bash
python testing.py --dataset_dir training_data_py --model_path ./fine_tuned_model
```

## 🔧 Usage

### Basic Conflict Resolution

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

### Model Training

```python
from training import prepare_training_data, train_model

# Prepare training data
conflict_dirs = get_conflict_dirs("training_data_py")
training_data = prepare_training_data(conflict_dirs)

# Train model with LoRA
train_model(training_data, output_dir="./models/conflict_resolver")
```

## 📊 Model Performance

The system evaluates model performance using:

- **Similarity Score**: Measures how closely the generated resolution matches the expected outcome
- **Inference Time**: Tracks resolution speed for performance optimization
- **Resolution Quality**: Assesses syntactic correctness and logical coherence

### Supported Models

- **Qwen/Qwen2.5-Coder-0.5B** (Primary model)
- Custom fine-tuned variants with LoRA adapters
- Extensible to other causal language models

## 🛠️ Configuration

Key configuration parameters in each module:

```python
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
MAX_TOKENS = 32768
```

### Training Configuration

- **LoRA Rank**: 16
- **Alpha**: 32
- **Dropout**: 0.1
- **Target Modules**: Query, Key, Value projections

## 📈 Dataset

The project works with structured conflict datasets containing:

- **Base version** (O.extension): Original common ancestor
- **Version A**: First branch changes
- **Version B**: Second branch changes  
- **Merged version**: Expected resolution
- **Conflict version** (M.extension): File with conflict markers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on the Transformers library by Hugging Face
- Uses Qwen2.5-Coder models for code understanding
- Inspired by research in automated program repair and code generation

## 📚 Research Context

This project was developed as part of a data science research initiative exploring the application of large language models to software engineering challenges. The work demonstrates how fine-tuned models can learn domain-specific patterns for conflict resolution.

## 🔮 Future Work

- Support for additional programming languages
- Integration with popular Git workflows
- Real-time IDE plugins
- Improved context understanding for complex conflicts
- Multi-file conflict resolution capabilities 