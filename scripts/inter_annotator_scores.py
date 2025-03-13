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
        df[f"{annotator1}-{annotator2}_scores"] = pd.Series(dtype=float)
        label_scores = {}
        all_ious = []  # Store all IOUs for computing average across labels

        for label in unique_labels:
            total_iou = 0
            valid_samples = 0

            # Compute sample-wise scores
            for idx, row in df.iterrows():
                if pd.isna(row[f"{annotator1}_annotations"]) or pd.isna(
                    row[f"{annotator2}_annotations"]
                ):
                    df.at[idx, f"{annotator1}-{annotator2}_scores"] = np.nan
                    continue

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
                    iou = 1.0
                    df.at[idx, f"{annotator1}-{annotator2}_scores"] = iou
                    total_iou += iou
                    valid_samples += 1
                    continue

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

                iou = compute_iou(vec1, vec2)
                df.at[idx, f"{annotator1}-{annotator2}_scores"] = iou
                total_iou += iou
                valid_samples += 1

            # Compute average for this label
            avg_iou = total_iou / valid_samples if valid_samples > 0 else np.nan
            label_scores[label] = avg_iou
            if not np.isnan(avg_iou):
                all_ious.append(avg_iou)

        # Compute average across all labels
        label_scores["overall_average"] = np.mean(all_ious) if all_ious else np.nan
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

    # Format results for both console and file output
    output_lines = []

    # Add information about samples and ignored labels
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
        # Print overall average first
        if np.isnan(label_scores["overall_average"]):
            output_lines.append("  Overall Average: No valid samples")
        else:
            output_lines.append(
                f"  Overall Average: {label_scores['overall_average']:.3f}"
            )
        output_lines.append("  Label-wise scores:")
        for label, score in label_scores.items():
            if label != "overall_average":
                if np.isnan(score):
                    output_lines.append(f"    {label}: No valid samples")
                else:
                    output_lines.append(f"    {label}: {score:.3f}")

    # Print to console
    print("\n".join(output_lines))

    # Save results
    excel_path = f"{args.output}.xlsx"
    txt_path = f"{args.output}.txt"

    # Save detailed sample-wise results to Excel
    result_df.to_excel(excel_path, index=False)

    # Save formatted results to text file
    with open(txt_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"\nDetailed sample-wise results saved to {excel_path}")
    print(f"Formatted results saved to {txt_path}")


if __name__ == "__main__":
    main()
