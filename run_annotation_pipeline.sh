# Step 1: Convert input dataset to desired format
# Converts the input JSONL file to a processed dataset, keeping only the "text" column
python -m inference_pipelines.dataset_conversion --input_path data_250_fixed.jsonl --output_path datasets/dataset_1 --columns_to_keep "text"

echo "Conversion complete"
# Step 2: Run LLM inference for data annotation
# Uses the configuration specified in inference_configs.yaml to run inference using the specified LLM
python -m inference_pipelines.inference_engine --section data_annotation --config inference_pipelines/inference_configs.yaml

echo "inference complete"

python -m inference_pipelines.dataset_conversion --input_path datasets/dataset_1 --output_path filtered_annotations/deephermes_3.jsonl

# Step 3: Extract span annotations from LLM responses
# Converts XML-style tagged responses into span annotations in the format [start, end, label]
python -m inference_pipelines.get_spans --section convert_llm_response_to_span_annotations --config inference_pipelines/inference_configs.yaml

echo "span conversion complete"
# Step 4: Filter and convert the annotated dataset
# Filters the dataset based on agreement scores and converts to final JSONL format
python -m inference_pipelines.dataset_conversion --input_path filtered_annotations/deephermes_3.jsonl --output_path filtered_annotations/deephermes_3.jsonl --filter-file agreement_scores_rest.xlsx --filter-field text

echo "dataset conversion complete"
# Step 5: Calculate inter-annotator agreement scores
# between annotations on a sample of 180 instances, excluding "non-fallacy" labels
python scripts/inter_annotator_scores.py filtered_annotations --num_samples 180 --ignore_labels "non-fallacy"
echo "inter annotator score calc complete"

