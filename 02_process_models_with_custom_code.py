import json
from pathlib import Path
from typing import Dict, List

import requests
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_file

from configuration import DatasetConfig
from utilities import setup_logging

load_dotenv()
logger = setup_logging(__name__)


# Directory for storing generated code locally
LOCAL_CODE_DIR = Path("custom_code")
LOCAL_CODE_DIR.mkdir(parents=True, exist_ok=True)

# UV script header
UV_SCRIPT_HEADER = """\
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers",
#     "torch",
# ]
# ///
"""


def estimate_model_vram(model_id: str, api) -> float:
    try:
        model_info = api.model_info(model_id)

        if model_info.safetensors is None:
            logger.warning(f"No safetensors info for {model_id}")
            return 0.0

        total_params = model_info.safetensors.total
        param_dtypes = list(model_info.safetensors.parameters.keys())

        primary_dtype = param_dtypes[0] if param_dtypes else "FP32"
        bytes_per_param = 2 if "BF16" in primary_dtype or "FP16" in primary_dtype else 4

        vram_gb = (total_params * bytes_per_param) / (1024**3)
        return round(vram_gb * 1.3, 2)  # add 30% overhead

    except Exception as e:
        logger.error(f"Failed to estimate VRAM for {model_id}: {e}")
        return 0.0


def fetch_notebook_content(model_id: str):
    url = f"https://huggingface.co/{model_id}.ipynb"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return json.loads(response.text)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch notebook for {model_id}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in notebook for {model_id}: {e}")
    return None


def extract_code_cells(notebook: dict) -> List[str]:
    code_snippets = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell_content = "".join(cell.get("source", [])).strip()
        if cell_content.startswith("#"):
            code_snippets.append(cell_content)
    return code_snippets


def wrap_code_snippet_for_execution(
    code_content: str,
    model_name: str,
    snippet_index: int,
    execution_dataset_id: str,
) -> str:
    exec_file = f"{model_name}_{snippet_index}.txt"
    lines = ["try:"]
    lines += [f"    {line}" for line in code_content.splitlines()]
    lines += [
        f"    with open('{exec_file}', 'w') as f:",
        f"        f.write('Everything was good in {exec_file}')",
        "except Exception as e:",
        f"    with open('{exec_file}', 'w') as f:",
        "        import traceback",
        "        traceback.print_exc(file=f)",
        "finally:",
        "    from huggingface_hub import upload_file",
        "    upload_file(",
        f"        path_or_fileobj='{exec_file}',",
        f"        repo_id='{execution_dataset_id}',",
        f"        path_in_repo='{exec_file}',",
        "        repo_type='dataset',",
        "    )",
    ]
    return "\n".join(lines), exec_file


def sanitize_model_name(model_id: str) -> str:
    return "_".join(model_id.split("/"))


def get_hf_dataset_url(dataset_id: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{dataset_id}/raw/main/{filename}"


def process_notebook_to_scripts(model_id: str, config: DatasetConfig) -> List[str]:
    notebook = fetch_notebook_content(model_id)
    if not notebook:
        return []

    code_snippets = extract_code_cells(notebook)
    if not code_snippets:
        logger.error(f"No code cells found in notebook for {model_id}")
        return []

    processed_scripts = []
    model_name = sanitize_model_name(model_id)

    for idx, snippet in enumerate(code_snippets):
        wrapped_code, _ = wrap_code_snippet_for_execution(
            snippet, model_name, idx, config.custom_code_execution_files_dataset_id
        )
        processed_scripts.append(UV_SCRIPT_HEADER + "\n" + wrapped_code)

    logger.info(f"Processed {len(processed_scripts)} scripts for {model_id}")
    return processed_scripts


def process_model_entry(model_id: str, config: DatasetConfig, hf_api) -> Dict:
    model_name = sanitize_model_name(model_id)
    vram = estimate_model_vram(model_id, hf_api)
    scripts = process_notebook_to_scripts(model_id, config)

    code_urls = []
    execution_urls = []

    for idx, script in enumerate(scripts):
        if "⚠️ Type of model/library unknow" in script:
            code_urls.append("DO NOT EXECUTE")
            execution_urls.append("WAS NOT EXECUTED")
            continue

        local_py_filename = f"{model_name}_{idx}.py"
        local_execution_filename = f"{model_name}_{idx}.txt"
        local_path = LOCAL_CODE_DIR / local_py_filename

        local_path.write_text(script, encoding="utf-8")

        upload_file(
            repo_id=config.custom_code_py_files_dataset_id,
            path_or_fileobj=local_path,
            path_in_repo=local_py_filename,
            repo_type="dataset",
        )
        logger.info(f"Uploaded: {local_py_filename}")

        code_urls.append(
            get_hf_dataset_url(
                config.custom_code_py_files_dataset_id, local_py_filename
            )
        )
        execution_urls.append(
            get_hf_dataset_url(
                config.custom_code_execution_files_dataset_id, local_execution_filename
            )
        )

    return {
        "model_id": model_id,
        "vram": vram,
        "scripts": scripts,
        "code_urls": code_urls,
        "execution_urls": execution_urls,
    }


if __name__ == "__main__":
    hf_api = HfApi()
    ds_config = DatasetConfig()

    target_model_ids = load_dataset(
        ds_config.models_with_custom_code_dataset_id, split="train"
    )["custom_code"]

    dataset_records = {
        "model_id": [],
        "vram": [],
        "scripts": [],
        "code_urls": [],
        "execution_urls": [],
    }

    for model_id in target_model_ids:
        result = process_model_entry(model_id, ds_config, hf_api)
        for key in dataset_records:
            dataset_records[key].append(result[key])

    Dataset.from_dict(dataset_records).push_to_hub(ds_config.model_vram_code_dataset_id)
