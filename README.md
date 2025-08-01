# Synthetic Training Data Generator for LLM Fine-tuning

This repository contains a multi-modal synthetic data generation system that processes markdown documents and creates high-quality question-answer pairs for fine-tuning Large Language Models (LLMs). The system supports both **LMDeploy** and **OpenRouter** for inference, uses InternVL3 with advanced image processing capabilities, and generates diverse question types with varying difficulty levels.

## Overview

The project features a modular architecture with the following components:

### Core Script
- `generate_synthetic_data.py` - Main script using either LMDeploy or OpenRouter for sophisticated QA pair generation.

### Utilities Module
- `utilities/api_utils.py` - API integration for both LMDeploy and OpenRouter, with multi-modal support.
- `utilities/file_utils.py` - Advanced markdown processing and fact extraction
- `utilities/logger.py` - Colored logging system with configurable levels
- `utilities/check_image_urls.py` - Image URL validation tool for generated datasets

### Models Module
- `models/question_types.py` - Comprehensive question type definitions and difficulty classifications

## Key Features

### Multi-Modal Processing
- **Image Integration**: Automatically detects and processes image references ({{FIGURE_X.X}} format)
- **Multi-Modal Prompts**: Combines text and images in question generation

### Advanced Question Generation
- **15 Question Types**: From basic factual recall to complex multi-hop reasoning
- **Difficulty Levels**: Easy, Medium, and Hard classifications
- **Bulk Processing**: Efficient batch processing with configurable batch sizes
- **Context-Aware**: Uses TF-IDF and KNN to find related sections for enhanced context

### Robust Processing Pipeline
- **Retry Mechanism**: Configurable number of retries for failed sections
- **Error Handling**: Comprehensive error tracking and recovery
- **Memory Management**: Handles large documents with memory error protection
- **Metadata Support**: YAML frontmatter extraction and processing

### YAML Frontmatter Documentation

Markdown files should begin with optional YAML frontmatter for metadata. The system extracts this metadata for enhanced processing.

#### Format Requirements:
- Must be enclosed by `---` delimiters
- Must be valid YAML syntax
- Must appear at the very beginning of the file

#### Example:
```yaml
---
title: "Document Title"
description: "Brief document summary"
domain: "Topic domain"
source: "Source organization"
version: "Document version"
last_updated: "YYYY-MM-DD"
audience: "Intended readers"
keywords: ["list", "of", "keywords"]
---
```

#### Required Fields:
- `title`: Document title
- `description`: Brief summary of content

#### Recommended Fields:
- `domain`: Topic area (e.g., "Traffic Engineering")
- `source`: Originating organization
- `version`: Document version
- `keywords`: List of search terms

#### Processing Notes:
- All fields become available in the generated QA pairs
- Invalid YAML will be logged but won't stop processing
- Missing required fields will generate warnings

### Question Types Supported

| Question Type | Difficulty | Description |
|---------------|------------|-------------|
| FACTUAL_RECALL | Easy | Direct information retrieval |
| INFERENCE | Medium | Logical reasoning from text |
| MULTI_HOP_REASONING | Hard | Combining information across sections |
| APPLICATION | Medium | Scenario-based practical application |
| COMPARATIVE_ANALYSIS | Medium | Comparing concepts or approaches |
| CAUSE_EFFECT | Medium | Understanding causal relationships |
| SUMMARIZATION | Easy | Concise content summarization |
| HYPOTHETICAL | Hard | What-if scenario extensions |
| CRITICAL_ANALYSIS | Hard | Evaluating strengths and weaknesses |
| TECHNICAL_EXPLANATION | Medium | Simplifying complex concepts |
| PROCESS_WORKFLOW | Easy | Sequential process understanding |
| TRUE_FALSE_FILL_BLANK | Easy | Basic recall testing |
| CONTEXTUAL_AMBIGUITY | Hard | Context-dependent interpretation |
| FACT_BASED | Easy | Individual fact-based questions |
| SECTION_SUMMARY | Medium | Comprehensive section summaries |

## Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU(s) with sufficient VRAM 
- Turing and higher Nvidia GPU architecture
- Multi-GPU support available

### Python Dependencies
```bash
pip install lmdeploy[all]
pip install openai
pip install nest_asyncio
pip install scikit-learn
pip install python-dotenv
pip install colorama
pip install pyyaml
pip install timm
```

## Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/grhone/SyntheticDataGeneration
cd SyntheticDataGeneration
```

2. **Install dependencies**
```bash
pip install lmdeploy[all] openai nest_asyncio scikit-learn python-dotenv colorama pyyaml timm
```

3. **Configure environment variables**
Create a `.env` file in the root directory:
```env
# Inference Engine Configuration
INFERENCE_ENGINE=lmdeploy  # 'lmdeploy' or 'openrouter'
OPENROUTER_API_KEY=your_openrouter_api_key_here # Required if using openrouter

# Model Configuration
# For lmdeploy, use a local path or HuggingFace repo ID
# For openrouter, use a model name like 'openai/gpt-4o'
MODEL=OpenGVLab/InternVL3-8B-AWQ
MODEL_FORMAT=awq # Only for lmdeploy
NUM_GPUS=2 # Only for lmdeploy

# Processing Configuration
MARKDOWN_DOCS_DIR=markdown_docs
OUTPUT_DIR=output
MAX_RETRIES=50

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

4. **Prepare your data**
- Place markdown files in the `markdown_docs` directory
- Ensure images are organized in subdirectories matching document names
- Use `{{FIGURE_X.X}}` format for image references in markdown

## Usage

### Basic Usage
```bash
python generate_synthetic_data.py
```

### Environment Variables
The system is configured through environment variables in the `.env` file:

- `INFERENCE_ENGINE`: The inference engine to use, either `lmdeploy` or `openrouter` (default: `lmdeploy`).
- `OPENROUTER_API_KEY`: Your API key for OpenRouter (required if `INFERENCE_ENGINE` is `openrouter`).
- `MODEL`: Model identifier. For `lmdeploy`, this can be a local path or a HuggingFace repo ID. For `openrouter`, use the model name from their catalog (e.g., `openai/gpt-4o`).
- `MODEL_FORMAT`: Model format for `lmdeploy` (e.g., `awq`, `fp16`). Not used for `openrouter`.
- `NUM_GPUS`: Number of GPUs to use for `lmdeploy` (default: 1). Not used for `openrouter`.
- `MARKDOWN_DOCS_DIR`: Input directory for markdown files (default: markdown_docs)
- `OUTPUT_DIR`: Output directory for generated data (default: output)
- `MAX_RETRIES`: Maximum retry attempts for failed sections (default: 50)
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `LOG_FILE`: Optional path to log file (disables file logging if not set)

### Image URL Validation

After generating synthetic data, you can validate that all referenced image paths exist using the image validation utility:

```bash
# Validate a specific JSON file
python utilities/check_image_urls.py output/train-document-synthetic-data-2024-01-01_12-00-00.json

# Or use the utility programmatically
from utilities.check_image_urls import check_all_output_files
check_all_output_files("output")
```

**Features:**
- **Path Validation**: Checks if all referenced image files exist on the filesystem
- **Context Reporting**: Shows the question context for any invalid image paths
- **Batch Processing**: Can validate all JSON files in the output directory
- **Colored Output**: Uses colored terminal output for easy identification of issues

**Example Output:**
```
Found 2 invalid image path(s) in train-document-synthetic-data-2024-01-01_12-00-00.json:

- Image path: traffic_manual/missing_chart.png
  Context: According to the Traffic Signal Timing Manual, the split failure pattern shown in the chart...

- Image path: figures/nonexistent.jpg  
  Context: The pedestrian detection system requires proper calibration as demonstrated...
```

## Output Format

The system generates separate training and evaluation datasets:

### File Structure
- `train-{document-name}-synthetic-data-YYYY-MM-DD_HH-MM-SS.json` - Training data
- `eval-{document-name}-synthetic-data-YYYY-MM-DD_HH-MM-SS.json` - Evaluation data (section summaries)

### JSON Format
Each file contains question-answer pairs in this multi-modal format:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What are the key safety requirements for automated traffic signals?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "traffic_manual/FIGURE_5.2.png",
              "detail": "auto"
            }
          }
        ]
      },
      {
        "role": "assistant",
        "content": "<think>\nStep 1: Identify safety requirements from the document\nStep 2: Reference relevant regulations and standards\nStep 3: Provide comprehensive answer with examples\n</think>\n\nAccording to the Traffic Signal Timing Manual, automated traffic signals must meet several key safety requirements:\n\n1. **Fail-Safe Operation**: Systems must default to a safe state during malfunctions\n2. **Redundant Detection**: Multiple detection methods for vehicle and pedestrian presence\n3. **Minimum Timing Standards**: Adherence to MUTCD minimum green and clearance intervals\n..."
      }
    ],
    "question_type": "TECHNICAL_EXPLANATION",
    "difficulty": "medium"
  }
]
```

## How It Works

### Processing Pipeline

1. **Document Processing**
   - Reads markdown files with YAML frontmatter support
   - Extracts metadata and splits content into logical sections
   - Identifies and resolves image references

2. **Context Analysis**
   - Uses TF-IDF vectorization to find related sections
   - Applies K-Nearest Neighbors for contextual relevance
   - Extracts facts using advanced text processing

3. **Question Generation**
   - Processes each question type in bulk batches
   - Generates prompts with multi-modal content (text + images)
   - Uses sophisticated system prompts for consistent output

4. **Quality Assurance**
   - Implements retry mechanism for failed generations
   - Validates JSON output format
   - Tracks and reports processing statistics

5. **Output Generation**
   - Converts to training-ready format
   - Separates training and evaluation datasets
   - Adds metadata tags for question types and difficulty

### Advanced Features

#### Image Processing
- Automatically detects `{{FIGURE_X.X}}` references in markdown
- Resolves image paths across multiple possible locations
- Supports PNG, JPG, and JPEG formats
- Attaches relevant images to question contexts

#### Retry Logic
- Tracks failed sections and question types
- Implements exponential backoff for API calls
- Maintains processing statistics and error logs
- Ensures maximum data recovery

#### Memory Management
- Handles large documents with chunked processing
- Implements graceful degradation for memory constraints
- Provides detailed logging for troubleshooting

## Customization

### Adding New Question Types
1. Add new enum value to `models/question_types.py`
2. Define difficulty level in `QUESTION_TYPE_DIFFICULTY`
3. Implement prompt template in `generate_qa_pairs_bulk()`

### Modifying Processing Logic
- **Fact Extraction**: Modify `extract_facts_from_section()` in `utilities/file_utils.py`
- **Section Splitting**: Adjust `split_into_sections()` logic
- **Image Processing**: Customize `resolve_image_paths()` for different naming conventions

### Output Format Changes
- Modify `output_to_phi_format()` for different training frameworks

## Example Output

For a document about traffic signal timing, the system generates questions like:

**Multi-Modal Technical Question:**
- **Question**: "Analyze the ATSPM chart shown and explain what the split failure pattern indicates about signal timing efficiency."
- **Images**: `traffic_manual/FIGURE_5.2.png`
- **Answer**: Detailed technical analysis analysis 

**Multi-Hop Reasoning:**
- **Question**: "How do the pedestrian clearance requirements in Section 3 relate to the vehicle detection standards in Section 7?"
- **Answer**: Comprehensive explanation connecting multiple document sections

## Troubleshooting

### Common Issues
- **GPU Memory**: Reduce `NUM_GPUS` or use smaller model format
- **Image Not Found**: Check image path resolution and naming conventions
  - Use the image validation utility: `python utilities/check_image_urls.py <json_file>`
  - Verify image directory structure matches document names
  - Ensure image files have correct extensions (png, jpg, jpeg)
- **JSON Parsing Errors**: Review system prompt formatting and model responses
- **High Retry Rates**: Adjust batch sizes or model parameters

### Validation Tools
- **Image Validation**: Use `utilities/check_image_urls.py` to verify all image paths in generated datasets
- **Logging**: Set `LOG_LEVEL=DEBUG` in `.env` for detailed processing information
- **Output Verification**: Check generated JSON files for proper formatting and completeness

## Contributing

When contributing to this project:
1. Maintain the modular architecture
2. Add comprehensive logging for new features
3. Update question type enums and difficulty mappings
4. Test with various document formats and image configurations
5. Ensure backward compatibility with existing output formats
