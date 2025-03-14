import re
import json
import yaml
import argparse
import pandas as pd
from pathlib import Path


def load_config(config_path, section="convert_llm_response_to_span_annotations"):
    """
    Load the configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file
        section (str): The section to load from the config file

    Returns:
        dict: The configuration for the specified section
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if section not in config:
        raise ValueError(f"Section '{section}' not found in config file")

    return config[section]


def load_dataset(input_dataset_path):
    """Load the dataset from the specified path."""
    # Determine file extension to handle different formats
    ext = Path(input_dataset_path).suffix.lower()

    if ext == ".json":
        with open(input_dataset_path, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    elif ext == ".jsonl":
        return pd.read_json(input_dataset_path, lines=True)
    elif ext == ".csv":
        return pd.read_csv(input_dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def get_matching_spans(pattern, text):
    """
    Get the spans of the text that match the pattern.
    """
    return next(re.finditer(pattern, text), None)


def convert_text_to_span_annotations(text, labels):
    """
    Convert tagged text to span annotations.

    Args:
        text (str): Text with XML-style tags
        labels (list): List of valid label types (e.g. ["credibility_fallacy", ...])

    Returns:
        list: List of [start, end, label] spans
    """
    # Create a working copy of the text
    cleaned_text = text.replace("<labeled_text>", "").replace("</labeled_text>", "")

    # Initialize list to store spans
    spans = []

    # Process each label type
    for label in labels:
        working_text = cleaned_text
        # remove all other tags
        for current_label in labels:
            if label != current_label:
                working_text = working_text.replace(f"<{current_label}>", "").replace(
                    f"</{current_label}>", ""
                )

        # Create regex patterns for opening and closing tags
        open_tag = f"<{label}>"
        close_tag = f"</{label}>"

        # Find all occurrences of this label in the text
        pattern = f"{open_tag}(.*?){close_tag}"

        # Track offset for character positions as we remove tags
        offset = 0

        # Find all matches
        # for match in re.finditer(pattern, working_text):
        while match := get_matching_spans(pattern, working_text):
            # Get the full match including tags
            full_match = match.group(0)
            # Get just the content between tags
            content = match.group(1)
            # Calculate start and end positions in the clean text
            start_pos = match.start()
            # Add the content length to get the end position
            end_pos = start_pos + len(content)
            # Add span to the list
            spans.append([start_pos, end_pos, label])
            # Replace the tagged text with just the content in the working text
            working_text = working_text.replace(full_match, content, 1)

    # Return the spans
    return spans


def process_dataset(config):
    """Process the dataset according to the configuration."""
    # Load the dataset
    df = load_dataset(config["input_dataset_path"])

    # Get the column names from config
    gen_col = config["generation_column_name"]
    span_col = config["span_annotations_column_name"]
    labels = config["labels"]

    # Process each row
    for idx, row in df.iterrows():
        # Get the text with annotations
        annotated_text = row[gen_col]

        # Extract the labeled text from within <labeled_text> tags if present
        labeled_text_match = re.search(
            r"<labeled_text>(.*?)</labeled_text>", annotated_text, re.DOTALL
        )
        if labeled_text_match:
            annotated_text = labeled_text_match.group(1).strip()

        # Convert to spans
        spans = convert_text_to_span_annotations(annotated_text, labels)
        # Add to dataframe
        df.at[idx, span_col] = (spans,)

    # Save the processed dataset
    save_dataset(df, config["result_dataset_path"])

    return df


def save_dataset(df, output_path):
    """Save the dataset to the specified path."""
    # Determine file extension to handle different formats
    ext = Path(output_path).suffix.lower()

    if ext == ".json":
        df.to_json(output_path, orient="records", indent=2)
    elif ext == ".jsonl":
        df.to_json(output_path, orient="records", lines=True)
    elif ext == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def test():
    """
    Test function to verify the functionality of the script with a small sample dataset.
    """
    print("Running test function...")

    # Create a temporary directory for test files
    import tempfile
    import os

    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "test_input.json")
    output_path = os.path.join(temp_dir, "test_output.json")

    # Create a sample dataset with annotated text
    sample_data = [
        {
            "id": 5,
            "response": "<labeled_text>The politician's argument <logical_fallacy>relies on cherry-picked data</logical_fallacy> and <logical_fallacy>appeals to fear of economic collapse</logical_fallacy>.</labeled_text>",
        },
        {
            "id": 1,
            "response": "<labeled_text><credibility_fallacy>Why did Poland push for regime change in Ukraine, <logical_fallacy>during the Maidan, so hard?</credibility_fallacy></logical_fallacy></labeled_text>",
        },
        {
            "id": 2,
            "response": "<labeled_text>This statement contains <logical_fallacy>if we allow gay marriage, next people will want to marry animals</logical_fallacy> which is a slippery slope.</labeled_text>",
        },
        {
            "id": 3,
            "response": "<labeled_text>The politician's argument <logical_fallacy>relies on cherry-picked data</logical_fallacy> and <emotional_fallacy>appeals to fear of economic collapse</emotional_fallacy>.</labeled_text>",
        },
        {
            "id": 4,
            "response": "This response doesn't have labeled_text tags but has <credibility_fallacy>direct annotation tags</credibility_fallacy> in the text.",
        },
        {
            "id": 6,  # overlapping tags
            "response": "<labeled_text>The<emotional_fallacy>politician's argument <logical_fallacy>relies on cherry-picked data</logical_fallacy> and appeals to fear of economic collapse</emotional_fallacy>.</labeled_text>",
        },
    ]

    # Save the sample dataset
    with open(input_path, "w") as f:
        json.dump(sample_data, f)

    # Create a test config
    test_config = {
        "input_dataset_path": input_path,
        "result_dataset_path": output_path,
        "generation_column_name": "response",
        "span_annotations_column_name": "llm_annotations",
        "labels": ["credibility_fallacy", "logical_fallacy", "emotional_fallacy"],
    }

    # Process the sample dataset
    df = process_dataset(test_config)

    # Print the results
    print("\nTest Results:")
    print("-" * 50)
    for idx, row in df.iterrows():
        print(f"Example {idx+1}:")
        print(f"Original: {row['response']}")
        print(f"Processed: {row['llm_annotations']}")
        print("-" * 50)

    print(f"\nTest output saved to: {output_path}")
    print(f"You can inspect the file for the complete results.")

    # Return the test results
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert LLM text annotations to span annotations"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="inference_configs.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--section",
        type=str,
        default="convert_llm_response_to_span_annotations",
        help="Section in the config file to use",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, args.section)

    # Check if required config values are set
    if config["input_dataset_path"] is None:
        raise ValueError(
            "input_dataset_path must be specified in the config file or via command line"
        )
    if config["result_dataset_path"] is None:
        raise ValueError(
            "result_dataset_path must be specified in the config file or via command line"
        )

    # Process the dataset
    process_dataset(config)

    print(f"Conversion complete. Results saved to {config['result_dataset_path']}")


if __name__ == "__main__":
    # test()
    main()
