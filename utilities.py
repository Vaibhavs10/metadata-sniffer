import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

from slack_sdk import WebClient


class SlackMessageType(Enum):
    HEADER = 1
    DIVIDER = 2
    SECTION = 3


@dataclass
class SlackMessage:
    msg_type: str = SlackMessageType.SECTION
    text: str = "default"


def setup_logging(name):
    logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(name)
    return logger


def format_slack_message(text, msg_type):
    if msg_type == SlackMessageType.HEADER:
        return {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": text,
                "emoji": False,
            },
        }
    elif msg_type == SlackMessageType.SECTION:
        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text,
            },
        }

    elif msg_type == SlackMessageType.DIVIDER:
        return {
            "type": "divider",
        }

    else:
        raise NotImplementedError("This type is not implemented")


def send_slack_message(
    client: WebClient,
    channel_name: str,
    messages: List[SlackMessage],
    simple_text=None,
    parent_message_ts=None,
):
    if simple_text is not None:
        response = client.chat_postMessage(
            channel=channel_name,
            text=simple_text,
            thread_ts=parent_message_ts,
        )

    else:
        blocks = []
        for message in messages:
            blocks.append(
                format_slack_message(text=message.text, msg_type=message.msg_type)
            )

        response = client.chat_postMessage(
            channel=channel_name,
            blocks=blocks,
            text="Hello",
            thread_ts=parent_message_ts,
        )

    return response
