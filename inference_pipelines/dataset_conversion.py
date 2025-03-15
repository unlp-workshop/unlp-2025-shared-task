import argparse
import json
import yaml
import pandas as pd
import os
from pathlib import Path
from datasets import Dataset, load_from_disk


def load_config(config_path, section):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config[section]


def detect_format(path):
    """Detect the format of a file based on its path."""
    path = str(path).lower()
    if path.endswith(".jsonl"):
        return "jsonl"
    elif path.endswith(".csv"):
        return "csv"
    elif path.endswith(".xlsx"):
        return "excel"
    elif os.path.isdir(path) or not os.path.exists(path):
        # Assume it's a HF dataset if it's a directory or doesn't exist yet
        return "hf_dataset"
    else:
        raise ValueError(f"Unsupported format for path: {path}")


def load_data(input_path, input_format=None):
    """Load data from various formats."""
    if input_format is None:
        input_format = detect_format(input_path)

    if input_format == "jsonl":
        # Read JSONL file
        data = []
        with open(input_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    elif input_format == "csv":
        # Read CSV file
        df = pd.read_csv(input_path)
        return df.to_dict("records")

    elif input_format == "excel":
        # Read Excel file
        df = pd.read_excel(input_path)
        return df.to_dict("records")

    elif input_format == "hf_dataset":
        # Load HF dataset
        dataset = load_from_disk(input_path)
        return dataset.to_list()

    else:
        raise ValueError(f"Unsupported input format: {input_format}")


def save_data(data, output_path, output_format=None):
    """Save data to various formats."""
    if output_format is None:
        output_format = detect_format(output_path)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if output_format == "jsonl":
        # Save as JSONL
        with open(output_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Data saved to JSONL file: {output_path}")

    elif output_format == "csv":
        # Save as CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Data saved to CSV file: {output_path}")

    elif output_format == "excel":
        # Save as Excel
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)
        print(f"Data saved to Excel file: {output_path}")

    elif output_format == "hf_dataset":
        # Save as HF dataset
        dataset = Dataset.from_list(data)
        dataset.save_to_disk(output_path)
        print(f"Data saved as HuggingFace dataset: {output_path}")

    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def convert_dataset(
    input_path, output_path, columns_to_keep=None, filter_file=None, filter_field=None
):
    """Convert dataset from one format to another."""
    input_format = detect_format(input_path)
    output_format = detect_format(output_path)
    detected_filter_format = detect_format(filter_file)

    print(f"Converting from {input_format} to {output_format}...")

    # Load data from input format
    data = load_data(input_path, input_format)

    # Apply filtering if filter_file is provided
    if filter_file and filter_field:
        filter_data = load_data(filter_file, detected_filter_format)
        filter_values = set(
            item[filter_field] for item in filter_data if filter_field in item
        )
        data = [
            item
            for item in data
            if filter_field in item and item[filter_field] in filter_values
        ]

    if columns_to_keep is not None:
        data = [
            {k: v for k, v in item.items() if k in columns_to_keep} for item in data
        ]

    # Save data to output format
    save_data(data, output_path, output_format)

    print(f"Conversion completed: {input_path} → {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert between dataset formats (JSONL, CSV, Excel, HuggingFace)"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output dataset",
    )
    parser.add_argument(
        "--columns_to_keep",
        type=str,
        required=False,
        help="Columns to keep in the output dataset",
        default=None,
    )
    parser.add_argument(
        "--filter-file",
        type=str,
        required=False,
        help="Path to file containing filter values",
        default=None,
    )
    parser.add_argument(
        "--filter-field",
        type=str,
        required=False,
        help="Field name to use for filtering",
        default=None,
    )

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    columns_to_keep = args.columns_to_keep
    if columns_to_keep is not None and not isinstance(columns_to_keep, list):
        columns_to_keep = [columns_to_keep,]  # fmt: skip

    # Convert dataset
    convert_dataset(
        input_path, output_path, columns_to_keep, args.filter_file, args.filter_field
    )


if __name__ == "__main__":
    main()
