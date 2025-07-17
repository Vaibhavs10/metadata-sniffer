import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import upload_file
from slack_sdk import WebClient

from configuration import DatasetConfig, ExecuteCustomCodeConfig, SlackConfig
from utilities import SlackMessage, SlackMessageType, send_slack_message, setup_logging

load_dotenv()
logger = setup_logging(__name__)

GPU_VRAM_MAPPING = {
    "l4x1": 30,
    "a10g-large": 46,
}
VRAM_TO_GPU_MAPPING = dict(
    sorted({vram: gpu for gpu, vram in GPU_VRAM_MAPPING.items()}.items())
)
LOCAL_CODE_DIR = Path("execution")
LOCAL_CODE_DIR.mkdir(parents=True, exist_ok=True)


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
    client = WebClient(token=os.environ["SLACK_TOKEN"])
    config = ExecuteCustomCodeConfig()
    ds_config = DatasetConfig()
    slack_config = SlackConfig()

    ds = load_dataset(ds_config.model_vram_code_dataset_id, split="train")

    slack_message_report = {
        "models_with_no_safetensors": [],
        "models_with_greater_vram": [],
        "models_with_no_code_found": [],
    }
    models_executed_with_urls = {
        "model_id": [],
        "index": [],
        "job_url": [],
        "job_id": [],
        "execution_url": [],
    }

    for sample in ds:
        model_id = sample["model_id"]
        estimated_vram = sample["vram"]
        script_urls = sample["code_urls"]
        execution_urls = sample["execution_urls"]

        if estimated_vram == 0.0:
            # There were no safetensors or a problem to estimate the VRAM
            slack_message_report["models_with_no_safetensors"].append(model_id)
            continue

        selected_gpu = select_appropriate_gpu(estimated_vram, execution_urls, model_id)
        if selected_gpu is None:
            # No gpus were found to run
            slack_message_report["models_with_greater_vram"].append(model_id)
            continue

        for idx, script_url in enumerate(script_urls):
            if "DO NOT EXECUTE" in script_url:
                slack_message_report["models_with_no_code_found"].append(model_id)
                logger.info(f"Skipping Execution {model_id}, no code found")
                break

            launch_command = (
                f"hfjobs run --detach --secret HF_TOKEN={os.getenv('HF_TOKEN')} --flavor {selected_gpu} {config.docker_image} /bin/bash -c "
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

                match = re.search(config.pattern, stdout)

                if match:
                    job_id = match.group(1)
                    job_url = match.group(2)

                    models_executed_with_urls["model_id"].append(model_id)
                    models_executed_with_urls["index"].append(idx)
                    models_executed_with_urls["job_id"].append(job_id)
                    models_executed_with_urls["job_url"].append(job_url)
                    models_executed_with_urls["execution_url"].append(
                        execution_urls[idx]
                    )

                    logger.info(f"{job_url} for {model_id} {idx}")

            except Exception as e:
                logger.error(f"Error launching job for {model_id}: {e}")

    # 5: Send the updates to slack
    today = datetime.now().strftime("%Y-%m-%d")
    messages = [
        SlackMessage(
            text=f"Custom Code Report for {today}", msg_type=SlackMessageType.HEADER
        )
    ]
    send_slack_message(
        client=client, channel_name=slack_config.channel_name, messages=messages
    )

    for issue_type, models in slack_message_report.items():
        messages = [
            SlackMessage(
                text=f"*{' '.join(issue_type.split('_'))}*",
                msg_type=SlackMessageType.SECTION,
            )
        ]

        text = ""
        for model in models:
            model_id = model
            # Check for the slack restriction
            if len(text + f"* <https://huggingface.co/{model_id}|{model_id}>\n") > 2900:
                messages.append(
                    SlackMessage(text=text, msg_type=SlackMessageType.SECTION)
                )
                text = f"* <https://huggingface.co/{model_id}|{model_id}>\n"
            else:
                text += f"* <https://huggingface.co/{model_id}|{model_id}>\n"

        messages.append(SlackMessage(text=text, msg_type=SlackMessageType.SECTION))

        send_slack_message(
            client=client, channel_name=slack_config.channel_name, messages=messages
        )

    # Upload the executed models information
    Dataset.from_dict(models_executed_with_urls).push_to_hub(
        ds_config.models_executed_with_urls_dataset_id
    )
