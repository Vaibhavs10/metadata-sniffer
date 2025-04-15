import argparse
from huggingface_hub import HfApi, file_exists, upload_file
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

        self.hf_api = HfApi()
        self.data_folder = "trending_data"
        self.repo_id = "ariG23498/trending_models"
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

    def send_slack_message(
        self, message: Optional[str] = None, blocks: Optional[List[Dict]] = None
    ) -> None:
        if not self.slack_webhook_url:
            logger.warning("Slack webhook URL not provided, skipping message")
            return

        payload = {"text": message or "Trending Model Issues"}
        if blocks:
            payload["blocks"] = blocks

        try:
            response = requests.post(self.slack_webhook_url, json=payload)
            response.raise_for_status()
            logger.info("Slack message sent successfully")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack message: {e}")

    def fetch_trending_models(self) -> List[Any]:
        logger.info(f"Fetching top {self.limit} trending models")
        try:
            return self.hf_api.list_models(sort="trendingScore", limit=self.limit)
        except Exception as e:
            logger.error(f"Error fetching trending models: {e}")
            return []

    def check_model_metadata(self, model) -> Dict[str, Any]:
        """Check model metadata and return model issues and status.

        Returns:
            A dictionary containing model ID, seen status, and any issues found.
        """
        model_id = model.id
        result = {
            "id": model_id,
            "has_been_seen": False,
            "issues": [],
            "avocado_discussions": [],
        }

        # Check if model has a discussion tab
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

    def format_issues_block(self) -> Dict:
        blocks = []

        def add_section(title, model_ids):
            if not model_ids:
                return

            block_text = f"*{title}* (_{len(model_ids)} models_)"
            for model_id in model_ids:
                # Find the seen status for this model
                has_been_seen = self._has_model_been_seen(model_id)
                status_emoji = "✅" if has_been_seen else "🔴"

                # Create a clean URL without the emoji
                block_text += (
                    f"\n• <https://huggingface.co/{model_id}|{model_id}> {status_emoji}"
                )

                # Add discussions if available
                discussions = self._get_model_discussions(model_id)
                if discussions:
                    block_text += "\n  _Discussions:_"
                    for disc in discussions:
                        block_text += (
                            f"\n  → <{disc['url']}|{disc['title']}> by {disc['author']}"
                        )

            blocks.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": block_text}}
            )
            blocks.append({"type": "divider"})

        blocks.append(
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🔍 Trending Model Issues Report {self.today}",
                    "emoji": True,
                },
            }
        )

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

        return {"blocks": blocks}

    def _has_model_been_seen(self, model_id: str) -> bool:
        return self.models_seen_status.get(model_id, False)

    def _get_model_discussions(self, model_id: str) -> List[Dict]:
        return self.models_avocado_discussions.get(model_id, [])

    def process_models(self) -> None:
        trending_models = self.fetch_trending_models()

        if not trending_models:
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
            with open(self.output_file, "w") as f:
                json.dump(model_data, f, indent=2, default=str)
            logger.info(f"Model data saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving model data: {e}")

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

    def notify_issues(self) -> None:
        if any(self.problematic_models.values()):
            slack_blocks = self.format_issues_block()
            if self.debug:
                import pprint

                logger.info("Slack blocks that would be sent:")
                pprint.pp(slack_blocks)
            else:
                self.send_slack_message(blocks=slack_blocks["blocks"])
                logger.info("Sent issues notification to Slack")
        else:
            logger.info("No issues found with trending models")

    def run(self) -> None:
        logger.info("Starting trending models check")
        self.process_models()
        self.upload_to_hub()
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
