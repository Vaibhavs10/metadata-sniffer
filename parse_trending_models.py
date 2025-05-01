import argparse
from huggingface_hub import HfApi, file_exists, upload_file
from datasets import load_dataset
import json
from datetime import datetime
import requests
import os
from dotenv import load_dotenv
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

AVOCADO_AUTHORS = [
    "ariG23498",
    "reach-vb",
    "pcuenq",
    "burtenshaw",
    "dylanebert",
    "davanstrien",
    "merve",
    "sergiopaniego",
    "Steveeeeeeen",
    "ThomasSimonini",
    "nielsr",
]


class ModelChecker:
    def __init__(self, debug: bool = False, limit: int = 100):
        self.debug = debug
        self.limit = limit

        # Configurations for the ModelChecker
        self.hf_api = HfApi()
        self.data_folder = "trending_data"
        self.repo_id = "model-metadata/trending_models"
        self.custom_code_ds_id = "model-metadata/model-id-custom-code-check"
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.today = datetime.now().strftime("%Y-%m-%d")

        os.makedirs(self.data_folder, exist_ok=True)
        self.output_file = f"{self.data_folder}/{self.today}.json"

        self.problematic_models = {
            "models_with_no_library_name": [],
            "models_with_no_pipeline_tag": [],
            "models_with_custom_code": [],
            "models_with_no_discussion_tab": [],
        }

        # Dictionary to track which models have been seen by the team
        self.models_seen_status = {}

        # Dictionary to store Avocado team discussions for each model
        self.models_avocado_discussions = {}

        # We have a dataset of all the models whose code snippets have been tried
        # previously. We would not want to check them again, and notify the
        # avocado team about it.
        custom_code_check_ds = load_dataset(self.custom_code_ds_id)
        self.checked_custom_models = custom_code_check_ds["train"]["model_id"]
        self.custom_model_id_to_description = {
            key: value
            for key, value in zip(
                self.checked_custom_models, custom_code_check_ds["train"]["description"]
            )
        }

    def fetch_trending_models(self) -> List[Any]:
        logger.info(f"Fetching top {self.limit} trending models")
        try:
            return self.hf_api.list_models(sort="trendingScore", limit=self.limit)
        except Exception as e:
            logger.error(f"Error fetching trending models: {e}")
            return None

    def check_model_metadata(self, model) -> Dict[str, Any]:
        model_id = model.id
        result = {
            "id": model_id,
            "has_been_seen": False,
            "issues": [],
            "avocado_discussions": [],
        }

        # Ignore GGUFs
        if "gguf" in (model.tags or []):
            logger.info(f"Skipping {model_id} as it is GGUF")
            return result

        # Some models do not have a discussion tab and it does not make
        # sense for the avocado team to check such models
        try:
            discussions = list(self.hf_api.get_repo_discussions(model_id))

            # Check if any Avocado team member has participated in discussions
            avocado_discussions = []
            for discussion in discussions:
                if discussion.author in AVOCADO_AUTHORS:
                    avocado_discussions.append(
                        {
                            "title": discussion.title,
                            "author": discussion.author,
                            "url": f"https://huggingface.co/{model_id}/discussions/{discussion.num}",
                        }
                    )

            result["has_been_seen"] = len(avocado_discussions) > 0
            result["avocado_discussions"] = avocado_discussions

        except Exception as e:
            logger.warning(f"Error fetching discussions for model {model_id}: {e}")
            result["issues"].append("models_with_no_discussion_tab")
            return result

        if model.library_name is None:
            result["issues"].append("models_with_no_library_name")

        if model.pipeline_tag is None:
            result["issues"].append("models_with_no_pipeline_tag")

        if "custom_code" in (model.tags or []):
            result["issues"].append("models_with_custom_code")

        return result

    def _categorize_problematic_models(self, results: List[Dict[str, Any]]) -> None:
        # Reset problematic models dictionary to ensure a clean slate
        self.problematic_models = {
            "models_with_no_library_name": [],
            "models_with_no_pipeline_tag": [],
            "models_with_custom_code": [],
            "models_with_no_discussion_tab": [],
        }

        # We could have added the models to the dicstionaries while checking the model
        # but that might create a race condition. It is safer to get all the threads to
        # ouput the results and then work on them sequentially and add it into the state vars.
        self.models_seen_status = {}
        self.models_avocado_discussions = {}

        # Process all results and categorize issues
        for result in results:
            model_id = result["id"]
            self.models_seen_status[model_id] = result["has_been_seen"]
            self.models_avocado_discussions[model_id] = result.get(
                "avocado_discussions", []
            )

            for issue_type in result["issues"]:
                if issue_type in self.problematic_models:
                    self.problematic_models[issue_type].append(model_id)

    def upload_to_hub(self) -> bool:
        # Only upload the json dump once in a day.
        try:
            if not file_exists(
                repo_id=self.repo_id, filename=self.output_file, repo_type="dataset"
            ):
                upload_file(
                    repo_id=self.repo_id,
                    path_or_fileobj=self.output_file,
                    path_in_repo=self.output_file,
                    repo_type="dataset",
                    commit_message=f"Upload {self.output_file}",
                )
                upload_message = f"Uploaded trending models data: {self.today}"
                self.send_slack_message(message=upload_message)
                logger.info(
                    f"Successfully uploaded {self.output_file} to {self.repo_id}"
                )
                return True
            else:
                logger.info(f"File {self.output_file} already exists in {self.repo_id}")
                return False
        except Exception as e:
            logger.error(f"Error uploading to HuggingFace Hub: {e}")
            return False

    def process_models(self) -> None:
        trending_models = self.fetch_trending_models()

        if trending_models is None:
            logger.error("No trending models found. Exiting.")
            return

        logger.info(f"Processing {self.limit} trending models")

        model_data = []
        results = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_model = {
                executor.submit(self.check_model_metadata, model): model
                for model in trending_models
            }

            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Store model data regardless of issues
                    # This will be dumped into a json and posted to a HF dataset
                    model_data.append(model.__dict__)
                except Exception as e:
                    logger.error(f"Error processing model {model.id}: {e}")

        self._categorize_problematic_models(results)

        try:
            with (
                open(self.output_file, "w") as f
            ):  # This creates a new output file per day, so that we do not overwite
                json.dump(model_data, f, indent=2, default=str)
            logger.info(f"Model data saved to {self.output_file}")
            self.upload_to_hub()

        except Exception as e:
            logger.error(f"Error saving model data: {e}")

    def send_slack_message(
        self, message: Optional[str] = None, blocks: Optional[List[Dict]] = None
    ) -> None:
        if not self.slack_webhook_url:
            logger.warning("Slack webhook URL not provided, skipping message")
            return

        # Handle simple text messages
        if blocks is None:
            self._send_simple_message(message or "Trending Model Issues")
            return

        # For block-based messages, chunk if necessary and send
        self._send_chunked_blocks(blocks)

    def _send_simple_message(self, message: str) -> None:
        payload = {"text": message}
        try:
            response = requests.post(self.slack_webhook_url, json=payload)
            response.raise_for_status()
            logger.info("Slack message sent successfully")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack message: {e}")

    def _send_chunked_blocks(self, blocks: List[Dict]) -> None:
        MAX_BLOCKS_PER_MESSAGE = 45  # Leave room for header block
        MAX_TEXT_LENGTH = 2800  # Conservative limit for text length

        # First pass: divide blocks into chunks
        chunks = []
        current_chunk = []
        current_text_length = 0

        for block in blocks:
            # Calculate the text length of this block
            block_text_length = 0
            if block.get("type") == "section" and block.get("text", {}).get("text"):
                block_text_length = len(block["text"]["text"])

            # Check if adding this block would exceed limits
            if (
                len(current_chunk) >= MAX_BLOCKS_PER_MESSAGE
                or (current_text_length + block_text_length) > MAX_TEXT_LENGTH
            ):
                # Current chunk is full, store it and start a new one
                if current_chunk:  # Only append non-empty chunks
                    chunks.append(current_chunk)
                current_chunk = [block]
                current_text_length = block_text_length
            else:
                # Add to current chunk
                current_chunk.append(block)
                current_text_length += block_text_length

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)

        total_chunks = len(chunks)

        # Send each chunk with appropriate headers
        for i, chunk in enumerate(chunks, 1):
            self._send_block_chunk(chunk, chunk_number=i, total_chunks=total_chunks)

    def _send_block_chunk(
        self, chunk: List[Dict], chunk_number: int, total_chunks: int
    ) -> None:
        payload = {"blocks": chunk}
        try:
            response = requests.post(self.slack_webhook_url, json=payload)
            response.raise_for_status()
            logger.info(
                f"Sent chunk {chunk_number}/{total_chunks} of Slack blocks message"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack blocks chunk {chunk_number}: {e}")

    def format_issues_block(self) -> Dict:
        blocks = []

        def add_section(title, model_ids):
            if not model_ids:
                return

            block_text = f"*{title}* (_{len(model_ids)} models_)"
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": block_text},
                }
            )

            current_block_text = ""

            for model_id in model_ids:
                # Find the seen status for this model
                has_been_seen = self.models_seen_status.get(model_id, False)
                status_emoji = "✅" if has_been_seen else "🔴"

                custom_model_snippet_check = False
                if title == "🧑‍💻 Models with Custom Code":
                    # Check if the model snippets were checked
                    custom_model_snippet_check = model_id in self.checked_custom_models
                    status_emoji = "✅" if custom_model_snippet_check else "🔴"

                # Format the model line
                model_line = (
                    f"\n• <https://huggingface.co/{model_id}|{model_id}> {status_emoji}"
                )

                # Add discussions if available
                discussions = self.models_avocado_discussions.get(model_id, [])
                for discussion in discussions:
                    model_line += f"\n\t → <{discussion['url']}|{discussion['title']}> by {discussion['author']}"

                if custom_model_snippet_check:
                    description = self.custom_model_id_to_description[model_id]
                    model_line += f"\n\t → Code was checked: {description}"

                current_block_text += model_line

            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": current_block_text},
                }
            )

            blocks.append({"type": "divider"})

        add_section(
            "📚 Models without a Library Name",
            self.problematic_models["models_with_no_library_name"],
        )
        add_section(
            "🏷️ Models without a Pipeline Tag",
            self.problematic_models["models_with_no_pipeline_tag"],
        )
        add_section(
            "🧑‍💻 Models with Custom Code",
            self.problematic_models["models_with_custom_code"],
        )
        add_section(
            "⛔️ Models without Discussion Tab",
            self.problematic_models["models_with_no_discussion_tab"],
        )

        return blocks

    def notify_issues(self) -> None:
        if any(self.problematic_models.values()):
            slack_blocks = self.format_issues_block()
            if self.debug:
                import pprint

                logger.info("Slack blocks that would be sent:")
                pprint.pp(slack_blocks)
            else:
                # Check if any section is too large
                total_size = sum(len(str(block)) for block in slack_blocks)
                if total_size > 3000:  # Conservative limit
                    logger.info(
                        f"Large payload detected ({total_size} chars), sending in chunks"
                    )

                self.send_slack_message(blocks=slack_blocks)
                logger.info("Sent issues notification to Slack")
        else:
            logger.info("No issues found with trending models")

    def run(self) -> None:
        logger.info("Starting trending models check")
        self.process_models()
        self.notify_issues()
        logger.info("Trending models check completed")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parses and validates the metadata of today's trending models"
    )

    parser.add_argument(
        "-d", "--debug", action="store_true", help="Run the script in debug mode"
    )

    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=100,
        help="Limit the number of trending models to fetch (default: 100)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    checker = ModelChecker(debug=args.debug, limit=args.limit)
    checker.run()


if __name__ == "__main__":
    main()
