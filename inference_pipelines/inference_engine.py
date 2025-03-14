import argparse
import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import yaml
from datasets import Dataset, DatasetDict, load_from_disk
from dotenv import load_dotenv
import pandas as pd
from tqdm.auto import tqdm
from inference_pipelines.prompts.annotation_guidelines import ANNOTATION_GUIDELINES
from inference_pipelines.prompts.annotation_prompt import ANNOTATION_PROMPT
from inference_pipelines.prompts.sample_annotations import FEW_SHOT_EXAMPLES

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/inference_engine.log"),
    ],
)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Import InferenceClient from utils
try:
    from inference_pipelines.utils import InferenceClient
except ImportError:
    logger.error(
        "Failed to import InferenceClient from utils. Make sure utils.py is in the path."
    )
    raise

# ===== Prompt Selection System =====


def select_prompt(prompt_to_use: str, text: str, **kwargs) -> List[Dict[str, str]]:
    """
    Select and call the appropriate prompt creation function based on the prompt_to_use parameter.

    Args:
        prompt_to_use: The name of the prompt creation function to use
        text: The input text to include in the prompt
        **kwargs: Additional arguments to pass to the prompt creation function

    Returns:
        List of message dictionaries in the format expected by the LLM
    """
    # Dictionary mapping prompt names to functions
    prompt_functions = {
        "annotation_prompt": annotation_prompt,
        # add more prompts here
    }

    if prompt_to_use not in prompt_functions:
        logger.warning(
            f"Prompt '{prompt_to_use}' not found. Using simple_prompt as fallback."
        )
        prompt_to_use = "simple_prompt"

    return prompt_functions[prompt_to_use](text, **kwargs)


def annotation_prompt(text: str, **kwargs) -> List[Dict[str, str]]:
    """
    Create an annotation prompt with the guidelines and prompt template.

    Args:
        text: The input text to include in the prompt
        **kwargs: Additional arguments (unused in this function)

    Returns:
        List with a single user message dictionary
    """
    prompt_template = ANNOTATION_PROMPT

    # replace text placeholder with text
    content = prompt_template.replace("{{TEXT}}", text)

    # replace guidelines placeholder with guidelines
    content = content.replace("{{GUIDELINES}}", ANNOTATION_GUIDELINES)

    # replace few shot examples placeholder with few shot examples
    content = content.replace("{{FEW_SHOT_EXAMPLES}}", FEW_SHOT_EXAMPLES)

    return [{"role": "user", "content": content}]


# ===== Inference Task Classes =====


@dataclass
class InferenceTask:
    """Class representing a single inference task."""

    row_idx: int
    message: List[dict]
    retry_count: int = 0
    permanently_failed: bool = False


@dataclass
class InferenceResult:
    """Class representing the result of an inference task."""

    generation: str
    token_count: int


class ResultTracker:
    """Class for tracking inference results and progress."""

    def __init__(self, total_tasks: int, config: dict):
        self.results: Dict[int, Optional[Tuple[str, int]]] = {}
        self.total_tasks = total_tasks
        self.processed_tasks = 0
        self.failed_tasks: List[InferenceTask] = []
        self.start_time = time.time()
        self.generation_column = config["generation_column_name"]
        self.token_count_column = f"{config['generation_column_name']}_token_count"
        self.pbar = tqdm(total=total_tasks, desc="Processing inferences", unit="task")
        logging.info(f"Initialized tracker with {total_tasks} tasks to process")

    def add_result(self, row_idx: int, completion: Optional[dict]) -> None:
        """Add a result for a row."""
        if completion is None:
            return

        self.results[row_idx] = (
            completion["generations"],
            completion["generated_token_count"],
        )

    def add_failed_task(self, task: InferenceTask):
        """Add a failed task to the tracker."""
        if not task.permanently_failed:
            self.failed_tasks.append(task)
            logging.warning(f"Added failed task for row {task.row_idx} to retry queue")

    def log_progress(self):
        """Update progress tracking."""
        self.processed_tasks += 1
        self.pbar.update(1)

        if self.processed_tasks % 100 == 0:
            elapsed_time = time.time() - self.start_time
            progress = (self.processed_tasks / self.total_tasks) * 100
            remaining = (elapsed_time / self.processed_tasks) * (
                self.total_tasks - self.processed_tasks
            )
            self.pbar.set_postfix(
                {"Elapsed": f"{elapsed_time:.0f}s", "Remaining": f"{remaining:.0f}s"}
            )
            logging.info(
                f"Processed {self.processed_tasks}/{self.total_tasks} tasks "
                f"({progress:.1f}%). Elapsed: {elapsed_time:.0f}s. "
                f"Estimated remaining: {remaining:.0f}s"
            )

    def close(self):
        """Close progress bar and log final statistics."""
        self.pbar.close()
        elapsed_time = time.time() - self.start_time
        logging.info(f"\nInference Statistics:")
        logging.info(f"Total tasks processed: {self.total_tasks}")
        logging.info(f"Total time: {elapsed_time:.0f}s")
        if self.failed_tasks:
            logging.warning(f"Failed tasks: {len(self.failed_tasks)}")


# ===== Direct Inference Functions =====


async def process_inference(
    task: InferenceTask,
    inference_client: InferenceClient,
    result_queue: asyncio.Queue,
    tracker: Optional[ResultTracker] = None,
):
    """Process a single inference task with retries."""
    while task.retry_count < MAX_RETRIES:
        try:
            # Generate completion
            completion = await inference_client.generate_completion(task.message)
            # Put result in queue
            await result_queue.put((task.row_idx, completion))
            return

        except Exception as e:
            task.retry_count += 1
            if task.retry_count < MAX_RETRIES:
                logging.warning(
                    f"Error processing row {task.row_idx} "
                    f"(attempt {task.retry_count}/{MAX_RETRIES}): {str(e)}. Retrying..."
                )
                await asyncio.sleep(RETRY_DELAY * task.retry_count)
            else:
                task.permanently_failed = True
                if tracker:
                    tracker.add_failed_task(task)
                logging.error(
                    f"Failed to process row {task.row_idx} "
                    f"after {MAX_RETRIES} attempts: {str(e)}"
                )
                await result_queue.put((task.row_idx, None))


async def result_handler(
    ds: pd.DataFrame,
    result_queue: asyncio.Queue,
    total_tasks: int,
    config: dict,
):
    """Handle results from the queue and update the dataset."""
    tracker = ResultTracker(total_tasks, config)

    try:
        while tracker.processed_tasks < total_tasks:
            row_idx, completion = await result_queue.get()

            try:
                # Add result
                tracker.add_result(row_idx, completion)

                if completion is not None:
                    ds.at[row_idx, tracker.generation_column] = completion[
                        "generations"
                    ]
                    ds.at[row_idx, tracker.token_count_column] = completion[
                        "generated_token_count"
                    ]
                    logging.info(f"Completed row {row_idx}")

                tracker.log_progress()

            except Exception as e:
                logging.error(f"Error updating results for row {row_idx}: {str(e)}")
            finally:
                result_queue.task_done()
    finally:
        tracker.close()


async def worker(
    task_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    inference_client: InferenceClient,
    tracker: Optional[ResultTracker] = None,
):
    """Worker that processes tasks from the queue."""
    while True:
        try:
            task = await task_queue.get()
            if task is None:  # Poison pill to stop the worker
                task_queue.task_done()
                break

            await process_inference(task, inference_client, result_queue, tracker)
            task_queue.task_done()

        except Exception as e:
            logging.error(f"Worker encountered error: {str(e)}")
            task_queue.task_done()


def get_inference_tasks(ds: pd.DataFrame, config: dict) -> List[InferenceTask]:
    """Generate inference tasks for all rows."""
    tasks = []
    text_column = config["text_column_name"]
    prompt_name = config.get("prompt", "annotation_prompt")

    for row_idx in range(len(ds)):
        row = ds.iloc[row_idx]
        # Skip rows where text_column is None
        if pd.isna(row[text_column]):
            logging.info(f"Skipping row {row_idx} as {text_column} is None")
            continue

        # Create message using the prompt selection system
        message = select_prompt(
            prompt_name,
            row[text_column],
            **config.get("prompt_kwargs", {}),
        )

        tasks.append(InferenceTask(row_idx, message))

    return tasks


async def process_all_rows(
    ds: pd.DataFrame,
    inference_client: InferenceClient,
    config: dict,
    batch_size: int = 5,
):
    """Process all rows using a worker pool."""
    # Create queues
    task_queue = asyncio.Queue()
    result_queue = asyncio.Queue()

    # Create tasks for all rows
    tasks = get_inference_tasks(ds, config)
    total_tasks = len(tasks)
    logging.info(f"Created {total_tasks} inference tasks")

    # Create a tracker for the main processing phase
    tracker = ResultTracker(total_tasks, config)

    # Enqueue all tasks
    for task in tasks:
        await task_queue.put(task)

    # Add poison pills for workers
    for _ in range(batch_size):
        await task_queue.put(None)

    # Create worker tasks
    workers = [
        asyncio.create_task(worker(task_queue, result_queue, inference_client, tracker))
        for _ in range(batch_size)
    ]

    # Create result handler
    result_processor = asyncio.create_task(
        result_handler(ds, result_queue, total_tasks, config)
    )

    # Wait for all tasks to complete
    await asyncio.gather(
        task_queue.join(),
        result_queue.join(),
        *workers,
        result_processor,
    )

    return tracker.failed_tasks


async def retry_failed_tasks(
    ds: pd.DataFrame,
    inference_client: InferenceClient,
    failed_tasks: List[InferenceTask],
    config: dict,
    batch_size: int = 5,
):
    """Retry all failed tasks with a fresh retry count."""
    if not failed_tasks:
        return []

    logging.info(f"Starting retry phase for {len(failed_tasks)} failed tasks")

    # Create new queues for retry phase
    task_queue = asyncio.Queue()
    result_queue = asyncio.Queue()

    # Reset retry count and add tasks to queue
    for task in failed_tasks:
        task.retry_count = 0
        task.permanently_failed = False
        await task_queue.put(task)

    # Add poison pills for workers
    for _ in range(batch_size):
        await task_queue.put(None)

    # Create worker tasks
    workers = [
        asyncio.create_task(worker(task_queue, result_queue, inference_client, None))
        for _ in range(batch_size)
    ]

    # Create result handler
    result_processor = asyncio.create_task(
        result_handler(ds, result_queue, len(failed_tasks), config)
    )

    # Wait for all retry tasks to complete
    await asyncio.gather(
        task_queue.join(),
        result_queue.join(),
        *workers,
        result_processor,
    )

    # Return any tasks that still failed
    return [task for task in failed_tasks if task.permanently_failed]


# ===== Batch Inference Functions =====


def split_batch_data(batch_data: List[dict], num_splits: int) -> List[List[dict]]:
    """Split batch data into approximately equal parts."""
    if not batch_data:
        return []

    # Calculate size of each split
    split_size = math.ceil(len(batch_data) / num_splits)

    # Split the data
    return [
        batch_data[i : i + split_size] for i in range(0, len(batch_data), split_size)
    ]


def write_split_jsonl(
    split_data: List[dict], base_jsonl_path: str, split_idx: int
) -> str:
    """Write a split of batch data to a JSONL file."""
    # Create split file path
    split_path = base_jsonl_path.replace(".jsonl", f"-split{split_idx}.jsonl")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(split_path), exist_ok=True)

    # Write the split data
    with open(split_path, "w") as f:
        for entry in split_data:
            f.write(json.dumps(entry) + "\n")

    logging.info(f"Created split file {split_path} with {len(split_data)} tasks")
    return split_path


def create_batch_jsonl(ds: pd.DataFrame, config: dict) -> None:
    """Create JSONL file(s) for batch inference with OpenAI API."""
    tasks = get_inference_tasks(ds, config)
    batch_data = []

    # Get batch creation config
    batch_creation_config = config["inference_client"]["batch_inference_creation"]
    model_name = batch_creation_config.get("model")
    jsonl_path = batch_creation_config.get("jsonl_path")
    num_splits = batch_creation_config.get("num_splits", 1)

    if not model_name:
        raise ValueError("No model specified in batch_inference_creation config")

    for task_idx, task in enumerate(tasks):
        # Create batch request entry following OpenAI Batch API format
        custom_id = f"inference-{task.row_idx}"
        batch_entry = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": task.message,
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "max_tokens": config.get("max_tokens", 1024),
            },
        }
        batch_data.append(batch_entry)

    # Create directory if it doesn't exist
    batch_file = Path(jsonl_path)
    batch_file.parent.mkdir(parents=True, exist_ok=True)

    if num_splits > 1:
        # Split the batch data
        splits = split_batch_data(batch_data, num_splits)

        # Write each split to a separate file
        for idx, split_data in enumerate(splits):
            write_split_jsonl(split_data, str(batch_file), idx)

        logging.info(
            f"Created {len(splits)} split files from {len(batch_data)} total tasks"
        )
    else:
        # Write single JSONL file with index 0
        write_split_jsonl(batch_data, str(batch_file), 0)
        logging.info(
            f"Created batch JSONL file at {batch_file}-split0.jsonl with {len(batch_data)} tasks"
        )


def concatenate_batch_results(base_jsonl_path: str, num_splits: int) -> str:
    """Concatenate split batch result files into a single file."""
    base_path = Path(base_jsonl_path)
    result_path = str(base_path)

    # Create a set to track unique custom_ids
    seen_custom_ids = set()
    total_results = 0

    # Open the output file
    with open(result_path, "w") as outfile:
        # Process each split file
        for split_idx in range(num_splits):
            split_path = base_path.parent / base_path.name.replace(
                ".jsonl", f"-split{split_idx}.jsonl"
            )

            if not split_path.exists():
                logging.warning(f"Split file {split_path} not found")
                continue

            # Read and process each line from the split file
            with open(split_path, "r") as split_file:
                for line in split_file:
                    result = json.loads(line)
                    custom_id = result.get("custom_id")

                    # Check for duplicate custom_ids
                    if custom_id in seen_custom_ids:
                        logging.warning(f"Duplicate custom_id found: {custom_id}")
                        continue

                    seen_custom_ids.add(custom_id)
                    outfile.write(line)
                    total_results += 1

            # Optionally delete the split file after processing
            split_path.unlink()
            logging.info(f"Processed and removed split file: {split_path}")

    logging.info(f"Concatenated {total_results} results into {result_path}")
    return result_path


def process_batch_results(
    ds: pd.DataFrame, result_batch_jsonl_path: str, config: dict
) -> pd.DataFrame:
    """Process batch results from OpenAI Batch API and update the dataset."""
    # Get number of splits from config
    num_splits = config["inference_client"]["batch_inference_verification"].get(
        "num_splits", 1
    )

    # Always use concatenate_batch_results to handle both single and multiple splits
    result_batch_jsonl_path = concatenate_batch_results(
        result_batch_jsonl_path, num_splits
    )

    logging.info(f"Processing batch results from {result_batch_jsonl_path}")

    generation_column = config["generation_column_name"]
    token_count_column = f"{generation_column}_token_count"

    # Initialize generation columns if they don't exist
    if generation_column not in ds.columns:
        ds[generation_column] = None
        ds[token_count_column] = None

    # Statistics tracking
    stats = {"total": 0, "success": 0, "error": 0, "errors": {}}
    start_time = time.time()

    # Count total lines in file for progress tracking
    total_lines = sum(1 for _ in open(result_batch_jsonl_path))
    logging.info(f"Found {total_lines} results to process")

    # Create progress bar for batch processing
    pbar = tqdm(total=total_lines, desc="Processing batch results", unit="result")

    # Process each result in the batch results file
    with open(result_batch_jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id")

                if not custom_id:
                    logging.error(f"Missing custom_id in result: {result}")
                    continue

                # Extract row index from custom_id
                # Format: inference-{row_idx}
                import re

                match = re.match(r"inference-(\d+)", custom_id)
                if not match:
                    logging.error(f"Invalid custom_id format: {custom_id}")
                    continue

                row_idx = int(match.group(1))
                stats["total"] += 1

                # Check for errors in the response
                if result.get("error") or result["response"].get("error"):
                    error_type = (
                        result.get("error") or result["response"].get("error", {})
                    ).get("type", "UnknownError")
                    stats["error"] += 1
                    stats["errors"][error_type] = stats["errors"].get(error_type, 0) + 1
                    continue

                # Extract the model's response
                response_content = result["response"]["body"]["choices"][0]["message"][
                    "content"
                ].strip()

                # Update dataset with the generation
                ds.at[row_idx, generation_column] = response_content
                ds.at[row_idx, token_count_column] = result["response"]["body"][
                    "usage"
                ]["completion_tokens"]

                stats["success"] += 1

            except Exception as e:
                logging.error(
                    f"Error processing batch result (line {line_num}): {str(e)}"
                )
                stats["error"] += 1
                stats["errors"]["ProcessingError"] = (
                    stats["errors"].get("ProcessingError", 0) + 1
                )

            # Update progress bar
            pbar.update(1)

    # Close progress bar
    pbar.close()

    # Log final statistics
    elapsed_time = time.time() - start_time
    logging.info("\nBatch Processing Results Statistics:")
    logging.info(f"Total processed: {stats['total']} in {elapsed_time:.0f} seconds")
    if stats["total"] > 0:
        logging.info(
            f"Successful generations: {stats['success']} ({stats['success']/stats['total']*100:.2f}%)"
        )
        logging.info(
            f"Error cases: {stats['error']} ({stats['error']/stats['total']*100:.2f}%)"
        )

    if stats["errors"]:
        logging.info("\nError Statistics:")
        for error_type, count in stats["errors"].items():
            logging.info(f"{error_type}: {count} occurrences")

    return ds


# ===== Main Function =====


def main():
    """Main function to run the inference engine."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference on an LLM")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="inference_pipelines/inference_configs.yaml",
        help="Config file to use from inference_configs.yaml (e.g., data_annotation)",
    )
    parser.add_argument(
        "--section",
        type=str,
        required=True,
        help="Config section to use from inference_configs.yaml (e.g., data_annotation)",
    )
    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Load config
    config_path = args.config
    with open(config_path, "r") as f:
        all_configs = yaml.safe_load(f)

    if args.section not in all_configs:
        raise ValueError(f"Config section '{args.section}' not found in {config_path}")

    config = all_configs[args.section]

    # Check required config values
    required_fields = [
        "input_dataset_path",
        "result_dataset_path",
        "text_column_name",
        "generation_column_name",
    ]

    for field in required_fields:
        if not config.get(field):
            raise ValueError(f"Required config field '{field}' is missing or null")

    # Load dataset
    dataset_path = config["input_dataset_path"]
    result_path = config["result_dataset_path"]

    try:
        ds = load_from_disk(dataset_path)
        # If dataset is a DatasetDict, use the first split
        if isinstance(ds, DatasetDict):
            split_name = next(iter(ds))
            ds = ds[split_name].to_pandas()
        else:
            ds = ds.to_pandas()
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {dataset_path}: {str(e)}")

    # Reset index to handle any duplicate indices
    ds = ds.reset_index(drop=True)

    # Initialize generation columns
    generation_column = config["generation_column_name"]
    token_count_column = f"{generation_column}_token_count"

    if generation_column not in ds.columns:
        ds[generation_column] = None
    if token_count_column not in ds.columns:
        ds[token_count_column] = None

    # Determine inference mode
    inference_client_type = config.get("inference_client_type")

    if inference_client_type == "batch_inference_creation":
        # Create batch JSONL file for OpenAI API batch processing
        create_batch_jsonl(ds, config)
        logging.info(
            "Batch JSONL file created successfully. Submit this file to the OpenAI Batch API."
        )

    elif inference_client_type == "batch_inference_verification":
        # Process batch results and update dataset
        jsonl_path = config["inference_client"]["batch_inference_verification"][
            "jsonl_path"
        ]
        ds = process_batch_results(ds, jsonl_path, config)

        # Convert back to Dataset and save
        ds = Dataset.from_pandas(ds)
        ds.save_to_disk(result_path)
        logging.info(f"Processed batch results and saved dataset to {result_path}")

    else:
        # Direct inference mode
        load_dotenv()

        # Get inference client config
        inference_client_configs = config["inference_client"][inference_client_type]

        # Get API key from environment variable if specified with env var name
        api_key_name = inference_client_configs.get("api_key")
        api_key = os.getenv(api_key_name) if api_key_name else None

        # Get other inference parameters
        api_base = inference_client_configs["service_url"]
        model_name = inference_client_configs["model"]
        temperature = config.get("temperature", 0.7)
        top_p = config.get("top_p", 0.9)
        max_tokens = config.get("max_tokens", 1024)
        batch_size = config.get("batch_size", 5)

        # Initialize the inference client
        inference_client = InferenceClient(
            api_key,
            api_base,
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            async_client=True,
        )

        # Run inference
        failed_tasks = asyncio.run(
            process_all_rows(
                ds,
                inference_client,
                config,
                batch_size=batch_size,
            )
        )

        # Retry failed tasks
        if failed_tasks:
            logging.info(f"Attempting to retry {len(failed_tasks)} failed tasks")
            permanently_failed = asyncio.run(
                retry_failed_tasks(
                    ds,
                    inference_client,
                    failed_tasks,
                    config,
                    batch_size=batch_size,
                )
            )

            # Log any permanently failed tasks
            if permanently_failed:
                logging.warning(
                    f"{len(permanently_failed)} tasks permanently failed after all retries"
                )
                for task in permanently_failed:
                    logging.warning(f"Permanently failed task - row: {task.row_idx}")

        # Convert back to Dataset and save
        ds = Dataset.from_pandas(ds)
        ds.save_to_disk(result_path)
        logging.info(f"Inference completed and dataset saved to {result_path}")


if __name__ == "__main__":
    main()
