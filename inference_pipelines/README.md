# Inference Pipelines

This repository contains scripts for running inference pipelines on text datasets, including dataset conversion, LLM inference, and span annotation extraction.

## Table of Contents
- [Dataset Conversion](#dataset-conversion)
- [Inference Configuration](#inference-configuration)
- [Inference Engine](#inference-engine)
- [Span Extraction](#span-extraction)
- [Additional Components](#additional-components)

## Dataset Conversion

The `dataset_conversion.py` script allows you to convert datasets between different formats (JSONL, CSV, Excel, HuggingFace).

### Usage

```bash
python -m inference_pipelines.dataset_conversion \
    --input_path /path/to/input/dataset \
    --output_path /path/to/output/dataset \
    --columns_to_keep column1 column2
```

### Parameters

- `--input_path`: Path to the input dataset
- `--output_path`: Path to the output dataset
- `--columns_to_keep` (optional): Specific columns to keep in the output dataset

### Supported Formats

The script automatically detects formats based on file extensions:
- `.jsonl` - JSONL format
- `.csv` - CSV format
- `.xlsx` - Excel format
- Directory path - HuggingFace dataset

## Inference Configuration

The `inference_configs.yaml` file contains configuration settings for different inference tasks.

### Configuration Sections

#### Data Annotation

```yaml
data_annotation:
  input_dataset_path: datasets/dataset_1
  result_dataset_path: datasets/dataset_1
  prompt: annotation_prompt
  text_column_name: "text"
  generation_column_name: "response"
  temperature: 0.70
  top_p: 0.90
  batch_size: 50
  inference_client_type: vllm
  inference_client:
    vllm:
      model: "NousResearch/Hermes-3-Llama-3.2-3B"
      service_url: "http://localhost:42003/v1"
      api_key: vLLM_API_KEY
```

#### Span Annotation Conversion

```yaml
convert_llm_response_to_span_annotations:
  input_dataset_path: datasets/dataset_1
  result_dataset_path: datasets/dataset_1
  generation_column_name: "response"
  span_annotations_column_name: "label"
  labels: ["credibility_fallacy", "logical_fallacy", "emotional_fallacy"]
```

## Inference Engine

The `inference_engine.py` script runs inference on text datasets using LLMs.

### Usage

```bash
python -m inference_pipelines.inference_engine \
    --config inference_pipelines/inference_configs.yaml \
    --section data_annotation
```

### Parameters

- `--config`: Path to the configuration file
- `--section`: Section in the configuration file to use

### Inference Modes

The script supports three inference modes:

1. **Direct Inference**: Processes data directly using an LLM API
2. **Batch Inference Creation**: Creates JSONL files for OpenAI Batch API
3. **Batch Inference Verification**: Processes results from OpenAI Batch API

### Adding New Prompts

To add a new prompt:

1. Create a new prompt file in the `prompts` directory
2. Add your prompt function to the `select_prompt` function in `inference_engine.py`:

```python
def select_prompt(prompt_to_use: str, text: str, **kwargs) -> List[Dict[str, str]]:
    # Dictionary mapping prompt names to functions
    prompt_functions = {
        "annotation_prompt": annotation_prompt,
        "your_new_prompt": your_new_prompt_function,  # Add your new prompt here
    }
```

3. Implement your prompt function:

```python
def your_new_prompt_function(text: str, **kwargs) -> List[Dict[str, str]]:
    """
    Create your custom prompt.
    
    Args:
        text: The input text to include in the prompt
        **kwargs: Additional arguments
        
    Returns:
        List with message dictionaries
    """
    # Your prompt implementation
    content = "Your prompt template with {{TEXT}} placeholder".replace("{{TEXT}}", text)
    
    return [{"role": "user", "content": content}]
```

## Span Extraction

The `get_spans.py` script extracts span annotations from LLM-generated text.

### Usage

```bash
python -m inference_pipelines.get_spans \
    --config inference_pipelines/inference_configs.yaml \
    --section convert_llm_response_to_span_annotations
```

### Parameters

- `--config`: Path to the configuration file
- `--section`: Section in the configuration file to use

### Annotation Format

The script expects LLM responses with XML-style tags:

```
<labeled_text>The politician's argument <logical_fallacy>relies on cherry-picked data</logical_fallacy> and <emotional_fallacy>appeals to fear of economic collapse</emotional_fallacy>.</labeled_text>
```

It converts these to span annotations in the format `[start, end, label]`.

## Additional Components

### Prompts Directory

The `prompts` directory contains prompt templates and guidelines:

- `annotation_prompt.py`: Template for annotation tasks
- `annotation_guidelines.py`: Guidelines for identifying fallacies
- `sample_annotations.py`: Few-shot examples for annotation

### Utils Module

The `utils.py` file contains utility functions, including the `InferenceClient` class for interacting with LLM APIs. This class handles:

- API authentication
- Request formatting
- Response processing
- Async operations
