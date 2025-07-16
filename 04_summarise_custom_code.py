from datasets import load_dataset
from dataclasses import dataclass
from slack_sdk import WebClient
import os
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


@dataclass
class SummarizerConfig:
    models_executed_with_urls_dataset_id = "model-metadata/models_executed_urls"
    channel_name = "#exp-slack-alerts"


def send_slack_block(block, channel):
    response = client.chat_postMessage(channel=channel, blocks=block)
    return response["ts"]


if __name__ == "__main__":
    config = SummarizerConfig()
    client = WebClient(token=os.environ["SLACK_TOKEN"])
    models_executed_with_urls = load_dataset(
        config.models_executed_with_urls_dataset_id, split="train"
    )

    today = datetime.now().strftime("%Y-%m-%d")
    block = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"HF Jobs Run Report for {today}",
                "emoji": False,
            },
        },
    ]
    _ = send_slack_block(block, config.channel_name)

    num_model_ids_executed = len(models_executed_with_urls["model_id"])
    for idx in range(num_model_ids_executed):
        model_id = models_executed_with_urls["model_id"][idx]
        job_id = models_executed_with_urls["job_id"][idx]
        job_url = models_executed_with_urls["job_url"][idx]
        index = models_executed_with_urls["index"][idx]
        execution_url = models_executed_with_urls["execution_url"][idx]

        block = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"<https://huggingface.co/{model_id}|{model_id}> ➡️ <{job_url}|Job Url>",
                },
            }
        ]
        parent_message_ts = send_slack_block(block, config.channel_name)

        execution_response = requests.get(execution_url).text
        client.chat_postMessage(
            channel=config.channel_name,
            text=execution_response,
            thread_ts=parent_message_ts,
        )
