import argparse
import json
import yaml
from pathlib import Path
from datasets import Dataset


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["dataset_creation"]


def jsonl_to_dataset(jsonl_path, result_dataset_path):
    """Convert JSONL file to Hugging Face dataset and save it."""
    # Read JSONL file
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # Convert to Dataset
    dataset = Dataset.from_list(data)

    # Save dataset
    dataset.save_to_disk(result_dataset_path)
    print(f"Dataset saved to {result_dataset_path}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL to Hugging Face dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="inference_pipelines/inference_configs.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    jsonl_path = config["jsonl_path"]
    result_dataset_path = config["result_dataset_path"]

    # Validate config
    if not jsonl_path:
        raise ValueError("jsonl_path not specified in config")
    if not result_dataset_path:
        raise ValueError("result_dataset_path not specified in config")

    # Ensure output directory exists
    Path(result_dataset_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert JSONL to dataset
    jsonl_to_dataset(jsonl_path, result_dataset_path)


if __name__ == "__main__":
    main()
