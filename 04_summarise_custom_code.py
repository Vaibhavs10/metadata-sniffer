import json
import logging
import os

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

MAX_BLOCKS_PER_MESSAGE = 45
MAX_TEXT_LENGTH_PER_BLOCK = 2900  # Slightly below Slack's 3000-char limit
MAX_PAYLOAD_SIZE = 3900  # Slightly below Slack's ~4000-char message limit


def send_slack_blocks(blocks, webhook_url):
    chunks = []
    current_chunk = []
    current_text_length = 0

    for block in blocks:
        # Calculate text length only for section blocks
        block_text_length = 0
        if block.get("type") == "section" and block.get("text", {}).get("text"):
            block_text_length = len(block["text"]["text"])

        # Start a new chunk if limits are exceeded
        if (
            len(current_chunk) >= MAX_BLOCKS_PER_MESSAGE
            or (current_text_length + block_text_length) > MAX_TEXT_LENGTH_PER_BLOCK
        ):
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [block]
            current_text_length = block_text_length
        else:
            current_chunk.append(block)
            current_text_length += block_text_length

    if current_chunk:
        chunks.append(current_chunk)

    for i, chunk in enumerate(chunks, 1):
        payload = {"blocks": chunk}
        payload_size = len(json.dumps(payload))

        if payload_size > MAX_PAYLOAD_SIZE:
            logger.warning(
                f"Chunk {i}/{len(chunks)} payload too large ({payload_size} chars), "
                "attempting to send simple text message"
            )
            send_simple_message(
                f"⚙️ Custom Code Execution Status: Chunk {i} too large to send",
                webhook_url,
            )
            continue

        try:
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"✅ Slack message chunk {i}/{len(chunks)} sent")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to send Slack message chunk {i}: {e}")
            # Log payload for debugging
            logger.debug(f"Failed chunk payload: {json.dumps(payload, indent=2)}")
            # Fallback to simple text message
            send_simple_message(
                f"⚙️ Custom Code Execution Status: Error sending chunk {i}", webhook_url
            )


def send_simple_message(message, webhook_url):
    """Send a simple text message as a fallback."""
    payload = {"text": message}
    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        logger.info("✅ Simple Slack message sent")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to send simple Slack message: {e}")


def format_execution_blocks(ds, client):
    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*⚙️ Custom Code Execution Status*"},
        },
        {"type": "divider"},
    ]

    # Group models by status
    successful = []
    no_code = []
    gpu_not_acquired = []
    errors = []

    for sample in ds:
        model_id = sample["model_id"]
        scripts = sample["scripts"]
        execution_urls = sample["execution_urls"]

        if "WAS NOT EXECUTED" in execution_urls:
            no_code.append((model_id, None, None))
            continue

        for idx, exec_url in enumerate(execution_urls):
            script = scripts[idx]
            try:
                execution_response = requests.get(exec_url).text
            except Exception as e:
                errors.append((model_id, "no logs", f"Error fetching log: {e}"))
                continue

            # Check for GPU not acquired
            if "No suitable GPU found for" in execution_response:
                gpu_not_acquired.append((model_id, f"<{exec_url}|log>", None))
                continue

            prompt = (
                f"For ```{script}``` the execution response is ```{execution_response}```.\n"
                f"Analyse the response. If the response is okay, just say `WORKS`, if it is about an error write a one line notification about the problem. Be very concise, not more than 30 tokens."
            )

            try:
                completion = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=[{"role": "user", "content": prompt}],
                )
                resp = completion.choices[0].message.content.strip()
                if resp == "WORKS":
                    successful.append((model_id, f"<{exec_url}|log>", resp))
                else:
                    errors.append((model_id, f"<{exec_url}|log>", resp))
            except Exception as e:
                errors.append((model_id, "no logs", f"Error from model inference: {e}"))

    # Format blocks for each status group, splitting large sections
    def add_status_section(title, items, emoji):
        if not items:
            return

        current_block_text = f"*{title}* (_{len(items)} models_)\n"
        block_count = 0

        for model_id, logs, details in items:
            line = f"• <https://huggingface.co/{model_id}|{model_id}> {emoji}"
            if logs:
                line += f" {logs}"
            if details and details != "WORKS":
                line += f" → {details}"
            line += "\n"

            # Check if adding this line would exceed the per-block limit
            if len(current_block_text) + len(line) > MAX_TEXT_LENGTH_PER_BLOCK:
                # Save current block
                blocks.append(
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": current_block_text.strip()},
                    }
                )
                block_count += 1
                # if block_count % 5 == 0:  # Add divider every 5 blocks for readability
                #     blocks.append({"type": "divider"})
                current_block_text = f"*{title} (continued)*\n{line}"
            else:
                current_block_text += line

        # Add the final block for this section
        if current_block_text.strip():
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": current_block_text.strip()},
                }
            )
            # blocks.append({"type": "divider"})

    add_status_section("✅ Successful Executions", successful, "✅")
    add_status_section("🚫 No Code to Execute", no_code, "🚫")
    add_status_section("🖥️ GPU Not Acquired", gpu_not_acquired, "🖥️")
    add_status_section("🚨 Execution Errors", errors, "🚨")

    return blocks


def main():
    slack_webhook_url = os.getenv("EXP_SLACK_WEBHOOK_URL")
    hf_token = os.getenv("HF_TOKEN")

    if not slack_webhook_url or not hf_token:
        raise ValueError("Missing required environment variables")

    ds = load_dataset("model-metadata/model_vram_code", split="train")
    client = InferenceClient(provider="together", api_key=hf_token)

    blocks = format_execution_blocks(ds, client)
    send_slack_blocks(blocks, slack_webhook_url)


if __name__ == "__main__":
    main()
