from huggingface_hub import HfApi, upload_file, list_repo_files
from datasets import Dataset, load_dataset
from pathlib import Path
import requests
import json
import os

# Constants
MODEL_IDS = [
    "Gen-Verse/MMaDA-8B-Base",
    "GSAI-ML/LLaDA-8B-Instruct",
    "pfnet/plamo-2-translate",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3-0324",
    "jinaai/jina-embeddings-v3",
]

DATASET_CODE_FILES = "model-metadata/custom-code-py-files"
DATASET_VRAM_CODE = "model-metadata/custom-vram-code"
CODE_DIR = Path("custom_code")
CODE_DIR.mkdir(exist_ok=True)

UV_HEADER = """\
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers",
#     "torch",
# ]
# ///
"""

GPU_VRAM = {
    "l4x1": 30,
    "a10g-small": 15,
    "a10g-large": 46,
}

VRAM_TO_GPU = dict(sorted({v: k for k, v in GPU_VRAM.items()}.items()))
DOCKER_IMAGE = "ghcr.io/astral-sh/uv:debian"


def get_notebook_code(model_id: str) -> str:
    url = f"https://huggingface.co/{model_id}.ipynb"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"[!] Failed to fetch notebook for {model_id}: {response.status_code}")
        return ""
    notebook = json.loads(response.text)
    lines = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            for line in cell.get("source", []):
                if line.strip().startswith("#"):
                    lines.append(line.rstrip())
    return "\n".join([UV_HEADER] + lines)


def write_and_upload_code(model_id: str, uploaded_files: set) -> str:
    file_name = "_".join(model_id.split("/")) + ".py"
    remote_path = f"custom_code/{file_name}"
    if remote_path in uploaded_files:
        print(f"[i] Skipping already uploaded file: {remote_path}")
        return remote_path

    code = get_notebook_code(model_id)
    if not code:
        return ""

    file_path = CODE_DIR / file_name
    file_path.write_text(code, encoding="utf-8")

    upload_file(
        repo_id=DATASET_CODE_FILES,
        path_or_fileobj=file_path,
        path_in_repo=remote_path,
        repo_type="dataset",
    )
    print(f"[✓] Uploaded: {remote_path}")
    return remote_path


def estimate_vram(model_id: str) -> float:
    api = HfApi()
    info = api.model_info(model_id)
    num_params = info.safetensors.total
    dtype = list(info.safetensors.parameters.keys())[0]
    bytes_per_param = 2 if dtype == "BF16" else 4
    return num_params * bytes_per_param / (1024 ** 3)


def upload_vram_metadata(vram_data: list[dict]):
    dataset = Dataset.from_list(vram_data)
    dataset.push_to_hub(DATASET_VRAM_CODE, split="train")


def select_gpu(vram_required: float) -> str:
    for vram, gpu in VRAM_TO_GPU.items():
        if vram_required < vram:
            return gpu
    return ""


def run_hfjobs():
    dataset = load_dataset(DATASET_VRAM_CODE)["train"]
    for row in dataset:
        model_id = row["model_id"]
        model_vram = row["vram"]
        model_name = "_".join(model_id.split("/"))
        script_url = f"https://huggingface.co/datasets/{DATASET_CODE_FILES}/raw/main/custom_code/{model_name}.py"

        gpu = select_gpu(model_vram)
        if not gpu:
            print(f"[!] Skipping {model_id}: VRAM too high")
            continue

        cmd = (
            f'hfjobs run --detach --flavor {gpu} {DOCKER_IMAGE} /bin/bash -c '
            f'"export HOME=/tmp && export USER=dummy && uv run {script_url}"'
        )
        os.system(cmd)
        print(f"[→] Launched job for {model_id}")


def main():
    uploaded_files = set(list_repo_files(DATASET_CODE_FILES, repo_type="dataset"))
    vram_metadata = []

    for model_id in MODEL_IDS:
        remote_path = write_and_upload_code(model_id, uploaded_files)
        if not remote_path:
            continue

        vram = estimate_vram(model_id)
        vram_metadata.append({"model_id": model_id, "vram": vram})
        print(f"[i] Estimated VRAM for {model_id}: {vram:.2f} GB")

    upload_vram_metadata(vram_metadata)
    run_hfjobs()


if __name__ == "__main__":
    main()
