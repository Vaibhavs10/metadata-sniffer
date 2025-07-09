import json
import logging
from pathlib import Path

import requests
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_file

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

HF_DATASET_CODE_REPO = "model-metadata/custom_code_py_files"
HF_DATASET_EXECUTION_REPO = "model-metadata/custom_code_execution_files"

LOCAL_CODE_DIR = Path("custom_code")
LOCAL_CODE_DIR.mkdir(parents=True, exist_ok=True)

# UV script header for Python dependencies
UV_SCRIPT_HEADER = """\
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers",
#     "torch",
# ]
# ///
"""


def estimate_model_vram(model_id: str) -> float:
    try:
        api = HfApi()
        model_info = api.model_info(model_id)

        if model_info.safetensors is None:
            logger.warning(f"No safetensors info for {model_id}")
            return 0.0

        total_params = model_info.safetensors.total
        param_dtypes = list(model_info.safetensors.parameters.keys())

        # Determine bytes per parameter based on dtype
        primary_dtype = param_dtypes[0] if param_dtypes else "FP32"
        bytes_per_param = 2 if "BF16" in primary_dtype or "FP16" in primary_dtype else 4

        # Calculate VRAM in GB
        vram_gb = (total_params * bytes_per_param) / (1024**3)

        # Add overhead for inference (typically 20-30%)
        vram_with_overhead = vram_gb * 1.3

        return round(vram_with_overhead, 2)

    except Exception as e:
        logger.error(f"Failed to estimate VRAM for {model_id}: {e}")
        return 0.0


def fetch_notebook_content(model_id: str):
    notebook_url = f"https://huggingface.co/{model_id}.ipynb"

    try:
        response = requests.get(notebook_url, timeout=30)
        response.raise_for_status()
        return json.loads(response.text)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch notebook for {model_id}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in notebook for {model_id}: {e}")
        return None


def extract_code_cells(notebook):
    code_snippets = []

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source_lines = cell.get("source", [])
        cell_content = "".join(source_lines).strip()

        # Only process cells that start with # (likely examples or comments)
        if cell_content and cell_content.startswith("#"):
            code_snippets.append(cell_content)

    return code_snippets


def wrap_code_with_error_handling(
    code_content: str, model_name: str, snippet_index: int
) -> str:
    execution_filename = f"{model_name}_{snippet_index}.txt"

    wrapped_lines = ["try:"]
    # Indent original code
    for line in code_content.splitlines():
        wrapped_lines.append(f"    {line}")

    # Add 'everything was good!' message in case of success
    wrapped_lines.extend(
        [
            f"    with open('{execution_filename}', 'w') as f:",
            f"        f.write('Everything was good in {model_name}_{snippet_index}')",
        ]
    )

    # Add exception handling block
    wrapped_lines.extend(
        [
            "except Exception as e:",
            f"    with open('{execution_filename}', 'w') as f:",
            "        import traceback",
            "        traceback.print_exc(file=f)",
        ]
    )

    # Add finally block for uploading the file to HuggingFace
    wrapped_lines.extend(
        [
            "finally:",
            "    from huggingface_hub import upload_file",
            "    upload_file(",
            f"        path_or_fileobj='{execution_filename}',",
            f"        repo_id='{HF_DATASET_EXECUTION_REPO}',",
            f"        path_in_repo='{execution_filename}',",
            "        repo_type='dataset',",
            "    )",
        ]
    )

    return "\n".join(wrapped_lines), execution_filename


def sanitize_model_name(model_id: str) -> str:
    """Convert model ID to filesystem-safe name."""
    return "_".join(model_id.split("/"))


def process_notebook_to_scripts(model_id: str):
    notebook = fetch_notebook_content(model_id)
    if not notebook:
        return []

    code_snippets = extract_code_cells(notebook)
    if not code_snippets:
        logger.info(f"No code cells found in notebook for {model_id}")
        return []

    model_name = model_id.split("/")[-1]
    processed_scripts = []
    processed_execution_filename = []

    for idx, snippet in enumerate(code_snippets):
        wrapped_code, execution_filename = wrap_code_with_error_handling(
            snippet, model_name, idx
        )
        full_script = UV_SCRIPT_HEADER + "\n" + wrapped_code
        processed_scripts.append(full_script)
        processed_execution_filename.append(execution_filename)

    logger.info(f"Processed {len(processed_scripts)} scripts for {model_id}")
    return processed_scripts, processed_execution_filename


def main():
    logger.info("Starting custom code processing pipeline")

    # Get all the models that have custom code snippets
    TARGET_MODEL_IDS = load_dataset(
        "model-metadata/models_with_custom_code", split="train"
    )["custom_code"]

    dataset_records = {
        "model_id": [],
        "vram": [],
        "scripts": [],
        "code_urls": [],
        "execution_urls": [],
    }

    for model_id in TARGET_MODEL_IDS:
        estimated_vram = estimate_model_vram(model_id)
        scripts, execution_filenames = process_notebook_to_scripts(model_id)
        code_urls = []
        execution_urls = []
        for idx, script in enumerate(scripts):
            if "⚠️ Type of model/library unknow" in script:
                code_urls.append("DO NOT EXECUTE")
                execution_urls.append("WAS NOT EXECUTED")
            else:
                # Write locally first
                local_filename = f"{model_id.split('/')[-1]}_{idx}.py"
                local_path = LOCAL_CODE_DIR / local_filename
                local_path.write_text(script, encoding="utf-8")

                upload_file(
                    repo_id=HF_DATASET_CODE_REPO,
                    path_or_fileobj=local_path,
                    path_in_repo=local_filename,
                    repo_type="dataset",
                )
                logger.info(f"Successfully uploaded: {local_filename}")
                code_urls.append(
                    f"https://huggingface.co/datasets/{HF_DATASET_CODE_REPO}/raw/main/{local_filename}"
                )
                execution_urls.append(
                    f"https://huggingface.co/datasets/{HF_DATASET_EXECUTION_REPO}/raw/main/{execution_filenames[idx]}"
                )

        dataset_records["model_id"].append(model_id)
        dataset_records["vram"].append(estimated_vram)
        dataset_records["scripts"].append(scripts)
        dataset_records["code_urls"].append(code_urls)
        dataset_records["execution_urls"].append(execution_urls)

    Dataset.from_dict(dataset_records).push_to_hub("model-metadata/model_vram_code")


if __name__ == "__main__":
    main()
