import logging
import os
import re
import subprocess
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import upload_file

load_dotenv()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GPU_VRAM_MAPPING = {
    "l4x1": 30,
    "a10g-large": 46,
}
VRAM_TO_GPU_MAPPING = dict(
    sorted({vram: gpu for gpu, vram in GPU_VRAM_MAPPING.items()}.items())
)
DOCKER_IMAGE_URL = "ghcr.io/astral-sh/uv:debian"

LOCAL_CODE_DIR = Path("execution")
LOCAL_CODE_DIR.mkdir(parents=True, exist_ok=True)

pattern = r"ID:\s*([a-zA-Z0-9]+)\s*View at:\s*(https://huggingface\.co/jobs/[^/]+/\1)"


def select_appropriate_gpu(vram_required: float, execution_urls, model_id):
    for available_vram, gpu_type in VRAM_TO_GPU_MAPPING.items():
        if vram_required < available_vram * 0.9:  # Leave 10% headroom
            return gpu_type

    for execution_url in execution_urls:
        file_name = execution_url.split("/")[-1]
        local_path = LOCAL_CODE_DIR / file_name
        local_path.write_text(
            f"No suitable GPU found for {model_id} | {vram_required:.2f} GB VRAM requirement",
            encoding="utf-8",
        )

        upload_file(
            path_or_fileobj=local_path,
            repo_id="model-metadata/custom_code_execution_files",
            path_in_repo=file_name,
            repo_type="dataset",
        )

    logger.warning(f"No suitable GPU found for {vram_required:.2f} GB VRAM requirement")
    return None


if __name__ == "__main__":
    ds = load_dataset("model-metadata/model_vram_code", split="train")

    for sample in ds:
        model_id = sample["model_id"]
        estimated_vram = sample["vram"]
        script_urls = sample["code_urls"]
        execution_urls = sample["execution_urls"]

        selected_gpu = select_appropriate_gpu(estimated_vram, execution_urls, model_id)
        if selected_gpu is None:  # no gpus were found to run
            continue

        for idx, script_url in enumerate(script_urls):
            if "DO NOT EXECUTE" in script_url:
                logger.info(f"Skipping Execution {model_id}, no code found")
                continue

            launch_command = (
                f"hfjobs run --detach --secret HF_TOKEN={os.getenv('HF_TOKEN')} --flavor {selected_gpu} {DOCKER_IMAGE_URL} /bin/bash -c "
                f'"export HOME=/tmp && export USER=dummy && uv run {script_url}"'
            )

            try:
                result = subprocess.run(
                    launch_command,
                    shell=True,
                    text=True,  # Ensures output is returned as string
                    capture_output=True,  # Captures stdout and stderr
                )

                exit_code = result.returncode
                stdout = result.stdout
                stderr = result.stderr

                if exit_code == 0:
                    logger.info(
                        f"Successfully launched job for {model_id} {idx} on {selected_gpu}"
                    )
                else:
                    logger.error(
                        f"Failed to launch job for {model_id}, exit code: {exit_code}"
                    )

                match = re.search(pattern, stdout)

                if match:
                    job_id = match.group(1)
                    job_url = match.group(2)
                    logger.info(f"{job_url} for {model_id} {idx}")

            except Exception as e:
                logger.error(f"Error launching job for {model_id}: {e}")
