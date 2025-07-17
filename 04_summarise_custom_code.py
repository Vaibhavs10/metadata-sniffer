import os
from datetime import datetime

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from slack_sdk import WebClient

from configuration import DatasetConfig, SlackConfig
from utilities import SlackMessage, SlackMessageType, send_slack_message, setup_logging

load_dotenv()
logger = setup_logging(__name__)


if __name__ == "__main__":
    ds_config = DatasetConfig()
    slack_config = SlackConfig()
    client = WebClient(token=os.environ["SLACK_TOKEN"])
    models_executed_with_urls = load_dataset(
        ds_config.models_executed_with_urls_dataset_id, split="train"
    )

    today = datetime.now().strftime("%Y-%m-%d")
    messages = [
        SlackMessage(
            text=f"HF Jobs Run Report for {today}", msg_type=SlackMessageType.HEADER
        )
    ]

    send_slack_message(
        client=client, channel_name=slack_config.channel_name, messages=messages
    )

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
        messages = [
            SlackMessage(
                text=f"<https://huggingface.co/{model_id}|{model_id}> ➡️ <{job_url}|Job Url>",
                msg_type=SlackMessageType.SECTION,
            )
        ]
        response = send_slack_message(
            client=client, channel_name=slack_config.channel_name, messages=messages
        )
        parent_message_ts = response["ts"]

        execution_response = requests.get(execution_url).text
        send_slack_message(
            client=client,
            channel_name=slack_config.channel_name,
            messages=None,
            simple_text=execution_response,
            parent_message_ts=parent_message_ts,
        )
