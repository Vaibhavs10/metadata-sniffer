import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, ModelInfo
from slack_sdk import WebClient

from configuration import DatasetConfig, ModelCheckerConfig, SlackConfig
from utilities import SlackMessage, SlackMessageType, send_slack_message, setup_logging

load_dotenv()
logger = setup_logging(__name__)


@dataclass
class ModelMetadataResult:
    id: str
    has_avocado_team_interaction: bool = False
    metadata_issues: list = field(default_factory=list)
    discussions_with_avocado_participation: list = field(default_factory=list)
    estimated_vram: float = 0.0


@dataclass
class AvocadoDiscussion:
    title: str
    author: str
    url: str


# Enum
class MetadataIssues(Enum):
    NO_LIBRARY_NAME = "no_library_name"
    NO_PIPELINE_TAG = "no_pipeline_tag"
    NO_DISCUSSION_TAB = "no_discussion_tab"
    WITH_CUSTOM_CODE = "with_custom_code"


# Util Methods
def analyze_model_metadata(
    huggingface_api: HfApi, model_info: ModelInfo
) -> ModelMetadataResult:
    """Analyzes metadata for a model, checking for issues and Avocado team interactions."""
    model_id = model_info.id
    metadata_result = ModelMetadataResult(id=model_id)

    # Ignore GGUFs
    if "gguf" in (model_info.tags or []):
        logger.info(f"Skipped {model_id} : GGUF")
        return metadata_result

    # Some models do not have a discussion tab and it does not make
    # sense for the avocado team to check such models
    try:
        discussions = list(huggingface_api.get_repo_discussions(model_id))
    except Exception:
        metadata_result.metadata_issues.append(MetadataIssues.NO_DISCUSSION_TAB)
        logger.info(f"Skipped {model_id} : No Discussion Tab")
        return metadata_result

    # Check if any Avocado team member has participated in discussions
    discussions_with_avocado = []
    for discussion in discussions:
        if discussion.author in config.avocado_team_members:
            discussions_with_avocado.append(
                AvocadoDiscussion(
                    title=discussion.title,
                    author=discussion.author,
                    url=f"https://huggingface.co/{model_id}/discussions/{discussion.num}",
                )
            )

    metadata_result.has_avocado_team_interaction = len(discussions_with_avocado) > 0
    metadata_result.discussions_with_avocado_participation = discussions_with_avocado

    # Check for the issues
    if model_info.library_name is None:
        metadata_result.metadata_issues.append(MetadataIssues.NO_LIBRARY_NAME)

    if model_info.pipeline_tag is None:
        metadata_result.metadata_issues.append(MetadataIssues.NO_PIPELINE_TAG)

    if "custom_code" in (model_info.tags or []):
        metadata_result.metadata_issues.append(MetadataIssues.WITH_CUSTOM_CODE)

    return metadata_result


if __name__ == "__main__":
    # Configuration
    ds_config = DatasetConfig()
    slack_config = SlackConfig()
    config = ModelCheckerConfig()
    huggingface_api = HfApi()
    client = WebClient(token=os.environ["SLACK_TOKEN"])

    # 1: Fetch the top N trending models
    trending_models = huggingface_api.list_models(
        sort="trendingScore", limit=config.num_trending_models
    )

    # 2: Process model metadata
    metadata_results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_model_info = {
            executor.submit(
                analyze_model_metadata, huggingface_api, model_info
            ): model_info
            for model_info in trending_models
        }

        for future in as_completed(future_to_model_info):
            model_info = future_to_model_info[future]
            try:
                result = future.result()
                metadata_results.append(result)
            except Exception as e:
                logger.error(f"Error processing model {model_info.id}: {e}")

    # 3: Categorize the models based on the issues
    models_by_issue_type = {issue.value: [] for issue in MetadataIssues}

    for metadata_result in metadata_results:
        for issue in metadata_result.metadata_issues:
            models_by_issue_type[issue.value].append(metadata_result)

    # 4: Upload the information of all models with custom code
    custom_code_dataset = {
        "custom_code": [],
    }
    for model_metadata_result in models_by_issue_type[
        MetadataIssues.WITH_CUSTOM_CODE.value
    ]:
        custom_code_dataset["custom_code"].append(model_metadata_result.id)
    Dataset.from_dict(custom_code_dataset).push_to_hub(
        ds_config.models_with_custom_code_dataset_id
    )

    # 5: Send the updates to slack
    today = datetime.now().strftime("%Y-%m-%d")
    messages = [
        SlackMessage(msg_type=SlackMessageType.DIVIDER),
        SlackMessage(
            text=f"Meta Data Report for {today}",
            msg_type=SlackMessageType.HEADER,
        ),
    ]
    send_slack_message(
        client=client, channel_name=slack_config.channel_name, messages=messages
    )

    for issue_type, models in models_by_issue_type.items():
        messages = [
            (
                SlackMessage(
                    text=f"*{' '.join(issue_type.split('_'))}*",
                    msg_type=SlackMessageType.SECTION,
                )
            )
        ]

        text = ""
        for model in models:
            if not model.has_avocado_team_interaction:
                model_id = model.id
                # Check for the slack restriction
                if (
                    len(text + f"* <https://huggingface.co/{model_id}|{model_id}>\n")
                    > 2900
                ):
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
