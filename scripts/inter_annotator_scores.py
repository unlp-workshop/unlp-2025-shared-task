#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Dict, Tuple, Set


def load_annotations(file_path: Path) -> List[Dict]:
    """Load annotations from a JSONL file."""
    annotations = []
    with open(file_path) as f:
        for line in f:
            annotations.append(json.loads(line))
    return annotations


def get_unique_labels(annotations_dict: Dict[str, List[Dict]]) -> Set[str]:
    """Extract unique labels from all annotations."""
    labels = set()
    for annotator_data in annotations_dict.values():
        for sample in annotator_data:
            for span in sample.get("label", []):
                if len(span) >= 3:  # Ensure span has start, end, and label
                    labels.add(span[2])
    return labels


def create_label_vectors(
    text: str, spans: List[List], max_len: int, label: str
) -> np.ndarray:
    """Convert text spans into a binary vector."""
    vector = np.zeros(max_len, dtype=int)
    for span in spans:
        if len(span) >= 3 and span[2] == label:  # Check if span has the target label
            start, end = span[0], span[1]
            if start < max_len:
                vector[start : min(end, max_len)] = 1
    return vector


def compute_iou(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Intersection over Union for two binary vectors."""
    intersection = np.sum(vec1 & vec2)
    union = np.sum(vec1 | vec2)
    return intersection / union if union > 0 else 0.0


def compute_agreement_scores(
    df: pd.DataFrame, annotators: List[str], unique_labels: Set[str]
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], pd.DataFrame]:
    """Compute inter-annotator agreement scores and sample-wise scores."""
    # Initialize results dictionary and score columns
    results = {}
    max_len = df["text"].str.len().max()

    # Compare each pair of annotators
    for annotator1, annotator2 in combinations(annotators, 2):
        label_scores = {}
        all_ious = []  # Store all IOUs for computing average across labels
        # Create empty DataFrame with columns for each label
        label_iou_df = pd.DataFrame(index=df.index, columns=list(unique_labels))
        for label in unique_labels:
            # Define function to compute IoU for each row
            def compute_row_iou(row):
                annotations1 = json.loads(row[f"{annotator1}_annotations"])
                annotations2 = json.loads(row[f"{annotator2}_annotations"])

                # Check if either annotator has this label in their annotations
                has_label1 = any(
                    span[2] == label for span in annotations1 if len(span) >= 3
                )
                has_label2 = any(
                    span[2] == label for span in annotations2 if len(span) >= 3
                )

                # If neither annotator has marked this label, count as perfect agreement
                if not has_label1 and not has_label2:
                    return 1.0

                vec1 = create_label_vectors(
                    row["text"],
                    annotations1,
                    max_len,
                    label,
                )
                vec2 = create_label_vectors(
                    row["text"],
                    annotations2,
                    max_len,
                    label,
                )

                return compute_iou(vec1, vec2)

            # Apply function to all rows
            label_iou_df[label] = df.apply(compute_row_iou, axis=1)
            label_scores[label] = label_iou_df[label].mean()

        # Calculate average IoU for all labels in each row
        df[f"{annotator1}-{annotator2}_scores"] = label_iou_df.mean(axis=1)
        # Compute average across all labels
        label_scores["overall_average"] = df[f"{annotator1}-{annotator2}_scores"].mean()
        results[(annotator1, annotator2)] = label_scores

    return results, df


def main():
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement scores"
    )
    parser.add_argument(
        "dir_path", type=str, help="Directory containing annotation JSONL files"
    )
    parser.add_argument(
        "--num_samples", type=int, help="Number of samples to process (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="agreement_scores",
        help="Base name for output files (default: agreement_scores)",
    )
    parser.add_argument(
        "--ignore_labels",
        type=str,
        nargs="+",
        help="Labels to ignore in the computation (optional)",
    )
    args = parser.parse_args()

    # Load and process annotations
    annotations_dict = {
        file_path.stem: load_annotations(file_path)
        for file_path in Path(args.dir_path).glob("*.jsonl")
    }

    # Create DataFrame with all annotations
    dfs = []
    for annotator, data in annotations_dict.items():
        df = pd.DataFrame(data)
        df["annotator"] = annotator
        df["annotations"] = df["label"].apply(json.dumps)
        dfs.append(df[["text", "annotator", "annotations"]])

    # Combine and pivot data
    combined_df = pd.concat(dfs, ignore_index=True)
    pivot_df = combined_df.pivot(
        columns="annotator", values="annotations", index="text"
    ).reset_index()

    # Rename columns and prepare for score computation
    annotators = [col for col in pivot_df.columns if col != "text"]

    # Drop rows where any annotator has empty annotations
    pivot_df = pivot_df.dropna(subset=annotators, how="any")

    pivot_df = pivot_df.rename(
        columns={col: f"{col}_annotations" for col in annotators}
    )

    # Select first n samples if specified
    if args.num_samples is not None:
        pivot_df = pivot_df.head(args.num_samples)
        print(f"\nUsing first {len(pivot_df)} samples for score computation")

    # Get unique labels and filter out ignored labels
    unique_labels = get_unique_labels(annotations_dict)
    if args.ignore_labels:
        unique_labels = unique_labels - set(args.ignore_labels)
        print(f"\nIgnoring labels: {', '.join(args.ignore_labels)}")

    # Compute scores
    scores, result_df = compute_agreement_scores(pivot_df, annotators, unique_labels)

    # Print results and save to files
    output_lines = []

    if args.num_samples is not None:
        output_lines.append(
            f"Using first {len(pivot_df)} samples for score computation"
        )

    if args.ignore_labels:
        output_lines.append(f"Ignoring labels: {', '.join(args.ignore_labels)}")

    output_lines.append("\nInter-annotator Agreement Scores (IoU):")
    output_lines.append("=======================================")

    for (annotator1, annotator2), label_scores in scores.items():
        output_lines.append(f"\n{annotator1} vs {annotator2}:")
        output_lines.append(
            f"  Overall average: {label_scores.pop('overall_average'):.3f}"
        )
        output_lines.append("  Label-wise scores:")
        for label, score in label_scores.items():
            output_lines.append(f"    {label}: {score:.3f}")

    print("\n".join(output_lines))

    excel_path = f"{args.output}.xlsx"
    txt_path = f"{args.output}.txt"

    result_df.to_excel(excel_path, index=False)

    with open(txt_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"\nDetailed sample-wise results saved to {excel_path}")
    print(f"Formatted results saved to {txt_path}")


if __name__ == "__main__":
    main()
