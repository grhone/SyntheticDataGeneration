# Synthetic Training Data Generator for LLM Fine-tuning

This repository contains a script to generate synthetic training data from markdown documents for fine-tuning Large Language Models (LLMs). The script processes markdown files and create question-answer pairs with reasoning steps, formatted for training.

## Overview

The project includes main script:
1. `generate_synthetic_data.py` - A script that uses lmdeploy with InternVL2-8B model to generate sophisticated QA pairs

The script:
- Processes markdown files from the `markdown_docs` directory
- Splits the content into meaningful sections
- Generates question-answer pairs with reasoning steps, including:
  - Factual recall questions
  - Inference questions
  - Multi-hop reasoning questions
  - Application/practical use questions
  - Comparative analysis questions
  - Cause-effect relationship questions
  - Summarization questions
  - Hypothetical scenario questions
  - Critical analysis questions
  - Technical explanation questions
  - Process/workflow questions
  - True/False and fill-in-the-blank questions
  - Contextual ambiguity resolution questions
- Formats the data into different Chain of Thought (CoT) reasoning levels
- Saves the results as JSON files in the `output` directory

## Requirements

- Python 3.6+
- Required packages: `json`, `random`, `re`, `lmdeploy[all]`, `nest_asyncio`, `argparse`, `time` 

Install the required packages:

```bash
pip install nest_asyncio
pip install lmdeploy[all]
```

## Usage

### Script with lmdeploy

Run the lmdeploy script with:

```bash
python generate_synthetic_data.py
```

#### lmdeploy Script Options

The lmdeploy script supports several command-line arguments:

- `--model`: Model to use (default: 'OpenGVLab/InternVL2-8B')
- `--max-tokens`: Maximum tokens for model response (default: 4096)
- `--temperature`: Temperature for model response (default: 0.7)
- `--no-llm`: Use simple heuristics instead of LLM (falls back to basic script functionality)

Example with custom parameters:

```bash
python generate_synthetic_data.py --temperature 0.8 --max-tokens 2048
```

## Output Format

The scripts generate two JSON files in the `output` directory:


1. `train-synthetic-data-YYYY-MM-DD_HH-MM-SS.json` - Training data
2. `validation-synthetic-data-YYYY-MM-DD_HH-MM-SS.json` - Evaluation data

Each file contains QA pairs in the following format:

```json
[
  [
    {"role": "user", "content": "What are the requirements for certification of compliance for Highly Automated Vehicles in Pennsylvania?"},
    {"role": "assistant", "content": "According to the Pennsylvania Department of Transportation guidelines for Highly Automated Vehicles, certification of compliance involves the following: ..."}
  ],
  [
    {"role": "user", "content": "What safety measures are required for Highly Automated Vehicles in Pennsylvania?"},
    {"role": "assistant", "content": "First, I need to identify what the Pennsylvania guidelines say about safety measures.\nAccording to the Pennsylvania Department of Transportation guidelines for Highly Automated Vehicles, safety measures involves the following: ..."}
  ]
]
```

The data includes four different formats for each QA pair:
1. No Chain of Thought (CoT)
2. One-step CoT
3. Two-step CoT
4. All-steps CoT

## How It Works

#### lmdeploy Script

The lmdeploy script uses the InternVL2-8B model through lmdeploy to generate QA pairs. It:
1. Sets up the lmdeploy pipeline with the specified model (or uses simple heuristics if --no-llm flag is set)
2. Processes each markdown file, splitting content into sections and extracting facts
3. For each section:
   - Generates fact-based QA pairs for each extracted fact
   - Creates a section summary QA pair
   - Generates additional questions of various types (inference, multi-hop, application, etc.) based on configured distributions
4. For LLM-generated questions:
   - Sends carefully crafted prompts to generate specific question types
   - Parses the JSON response
   - Falls back to simple heuristics if there's an error
5. Converts all QA pairs to different Chain of Thought formats (no CoT, 1-step, 2-step, full CoT)
6. Splits data into training and validation sets, saving as timestamped JSON files

## Example

If you have a markdown file about Pennsylvania's Highly Automated Vehicles regulations, the scripts will generate questions like:
- "What are the requirements for certification of compliance for Highly Automated Vehicles in Pennsylvania?"
- "What safety measures are required for Highly Automated Vehicles in Pennsylvania?"

With corresponding answers and reasoning steps.

## Customization

You can customize the scripts by modifying:
- `MARKDOWN_DOCS_DIR` - Directory containing markdown files
- `OUTPUT_DIR` - Directory for output JSON files
- `TRAIN_RATIO` - Ratio of training to evaluation data
