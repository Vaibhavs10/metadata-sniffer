import os
import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
import requests
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_file
from slack_sdk import WebClient

from configuration import SlackConfig, DatasetConfig
from utilities import setup_logging
from utilities import SlackMessage, SlackMessageType, send_slack_message, setup_logging

load_dotenv()
logger = setup_logging(__name__)


# Directory for storing generated code locally
LOCAL_CODE_DIR = Path("code")
LOCAL_CODE_DIR.mkdir(parents=True, exist_ok=True)

# UV script header
UV_SCRIPT_HEADER = """\
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "torchvision",
#     "transformers",
#     "accelerate",
#     "peft",
# ]
# ///
"""


@dataclass
class ModelCodeInfo:
    id: str
    scripts: str
    code_urls: str
    execution_urls: str
    estimated_vram: float


def estimate_model_vram(model_id: str, huggingface_api: HfApi) -> float:
    try:
        model_info = huggingface_api.model_info(model_id)

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
    "Get the notebook contents from model id"
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
    "From the notebook extract only the code cells"
    code_snippets = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell_content = "".join(cell.get("source", [])).strip()
        if cell_content.startswith("#"):
            code_snippets.append(cell_content)
    return code_snippets


# def wrap_code_snippet_for_execution(
#     code_content: str,
#     model_name: str,
#     snippet_index: int,
#     execution_dataset_id: str,
# ) -> str:
#     exec_file = f"{model_name}_{snippet_index}.txt"
#     lines = ["try:"]
#     lines += [f"    {line}" for line in code_content.splitlines()]
#     lines += [
#         f"    with open('{exec_file}', 'w') as f:",
#         f"        f.write('Everything was good in {exec_file}')",
#         "except Exception as e:",
#         f"    with open('{exec_file}', 'w') as f:",
#         "        import traceback",
#         "        traceback.print_exc(file=f)",
#         "finally:",
#         "    from huggingface_hub import upload_file",
#         "    upload_file(",
#         f"        path_or_fileobj='{exec_file}',",
#         f"        repo_id='{execution_dataset_id}',",
#         f"        path_in_repo='{exec_file}',",
#         "        repo_type='dataset',",
#         "    )",
#     ]
#     return "\n".join(lines), exec_file
def wrap_code_snippet_for_execution(
    code_content: str,
    model_name: str,
    snippet_index: int,
    execution_dataset_id: str,
) -> tuple[str, str]:
    """
    Wraps the provided code content in a safe try/except/finally block
    that logs results and uploads them to the Hugging Face dataset.

    Returns:
        wrapped_code (str): Executable Python script string
        exec_file (str): Local filename used for logs
    """
    exec_file = f"{model_name}_{snippet_index}.txt"

    template = """\
try:
{indented_code}
    with open('{exec_file}', 'w', encoding='utf-8') as f:
        f.write('Everything was good in {exec_file}')
except Exception as e:
    with open('{exec_file}', 'w', encoding='utf-8') as f:
        import traceback
        traceback.print_exc(file=f)
finally:
    from huggingface_hub import upload_file
    upload_file(
        path_or_fileobj='{exec_file}',
        repo_id='{execution_dataset_id}',
        path_in_repo='{exec_file}',
        repo_type='dataset',
    )
"""

    # indent user code by 4 spaces
    indented_code = "\n".join(
        "    {}".format(line) for line in code_content.splitlines()
    )

    wrapped_code = template.format(
        indented_code=indented_code,
        exec_file=exec_file,
        execution_dataset_id=execution_dataset_id,
    )

    return wrapped_code, exec_file


def sanitize_model_name(model_id: str) -> str:
    return "_".join(model_id.split("/"))


def get_hf_dataset_url(dataset_id: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{dataset_id}/raw/main/{filename}"


def process_notebook_to_scripts(
    model_name: str, model_id: str, ds_config: DatasetConfig
) -> List[str]:
    notebook = fetch_notebook_content(model_id)
    if not notebook:
        logger.error(f"No notebook found for {model_id}")
        return [], []

    code_snippets = extract_code_cells(
        notebook
    )  # Extract only code cells, a list of snippets
    if not code_snippets:
        logger.error(f"No code cells found in notebook for {model_id}")
        return [], []

    processed_scripts = []
    exec_files = []
    for idx, snippet in enumerate(code_snippets):
        wrapped_code, exec_file = wrap_code_snippet_for_execution(
            code_content=snippet,
            model_name=model_name,
            snippet_index=idx,
            execution_dataset_id=ds_config.code_execution_files_dataset_id,
        )
        processed_scripts.append(UV_SCRIPT_HEADER + "\n" + wrapped_code)
        exec_files.append(exec_file)

    logger.info(f"Processed {len(processed_scripts)} scripts for {model_id}")
    return processed_scripts, exec_files


def process_models(
    model_id: str, ds_config: DatasetConfig, huggingface_api: HfApi
) -> ModelCodeInfo:
    model_name = sanitize_model_name(
        model_id=model_id
    )  # sanitize the name as it will be used as file name
    estimated_vram = estimate_model_vram(
        model_id=model_id, huggingface_api=huggingface_api
    )  # get the estimated vram later used for deciding the machines to run the models on
    processed_scripts, exec_files = process_notebook_to_scripts(
        model_name=model_name, model_id=model_id, ds_config=ds_config
    )

    code_urls = []
    execution_urls = []

    for script, exec_file in zip(processed_scripts, exec_files):
        if (
            "⚠️ Type of model/library unknow" in script
        ):  # This is hardcoded as sometimes you will not be able to get the script from the URL
            code_urls.append("DO NOT EXECUTE")
            execution_urls.append("WAS NOT EXECUTED")
            continue

        local_py_filename = exec_file.replace(".txt", ".py")
        local_path = LOCAL_CODE_DIR / local_py_filename
        local_path.write_text(script, encoding="utf-8")

        upload_file(
            repo_id=ds_config.code_python_files_dataset_id,
            path_or_fileobj=local_path,
            path_in_repo=local_py_filename,
            repo_type="dataset",
        )
        logger.info(f"Uploaded: {local_py_filename}")

        code_urls.append(
            get_hf_dataset_url(
                dataset_id=ds_config.code_python_files_dataset_id,
                filename=local_py_filename,
            )
        )
        execution_urls.append(
            get_hf_dataset_url(
                dataset_id=ds_config.code_execution_files_dataset_id,
                filename=exec_file,
            )
        )

    return ModelCodeInfo(
        id=model_id,
        estimated_vram=estimated_vram,
        scripts=processed_scripts,
        code_urls=code_urls,
        execution_urls=execution_urls,
    )


if __name__ == "__main__":
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    dataset_config = DatasetConfig()
    client = WebClient(token=os.environ["SLACK_TOKEN"])
    slack_config = SlackConfig()

    trending_models_metadata_ds = load_dataset(
        dataset_config.trending_models_metadata_id, split="train"
    )
    # filter models without ggufs and discussion tabs
    targeted_models = trending_models_metadata_ds.filter(
        lambda x: not x["should_skip"]
    )["id"]

    # could have used the threadpooler, but this function works on files (which is async)
    dataset_rows = list()
    for model_id in targeted_models:
        model_code_info = process_models(
            model_id=model_id, ds_config=dataset_config, huggingface_api=huggingface_api
        )
        dataset_rows.append(asdict(model_code_info))

    model_code_info_ds = Dataset.from_list(dataset_rows)
    model_code_info_ds.push_to_hub(dataset_config.hf_jobs_url_dataset_id)
    send_slack_message(
        client=client,
        channel_name=slack_config.channel_name,
        simple_text=f"HF Jobs URL Dataset Uploaded to <https://huggingface.co/datasets/{dataset_config.hf_jobs_url_dataset_id}|{dataset_config.hf_jobs_url_dataset_id}>",
    )
