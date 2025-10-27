import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import List, Dict

from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, ModelInfo
from slack_sdk import WebClient

from configuration import ModelCheckerConfig, SlackConfig
from utilities import SlackMessage, SlackMessageType, send_slack_message, setup_logging

load_dotenv()
logger = setup_logging(__name__)


@dataclass
class ModelMetadataResult:
    id: str
    should_skip: bool = False
    metadata_issues: List[str] = field(default_factory=list)
    discussions_with_avocado_participation: List["AvocadoDiscussion"] = field(
        default_factory=list
    )
    estimated_vram: float = 0.0


@dataclass
class AvocadoDiscussion:
    title: str
    author: str
    url: str


class MetadataIssues(Enum):
    NO_LIBRARY_NAME = "no_library_name"
    NO_PIPELINE_TAG = "no_pipeline_tag"
    NO_DISCUSSION_TAB = "no_discussion_tab"
    WITH_GGUF = "with_gguf"


def _model_link_line(model_id: str) -> str:
    return f"* <https://huggingface.co/{model_id}|{model_id}>\n"


def _chunk_markdown(text_lines: List[str], max_len: int = 2900) -> List[str]:
    """Split lines into chunks under Slack section hard-limit."""
    chunks: List[str] = []
    buf = ""
    for line in text_lines:
        if len(buf) + len(line) > max_len:
            chunks.append(buf)
            buf = line
        else:
            buf += line
    if buf:
        chunks.append(buf)
    return chunks


def analyze_model_metadata(
    huggingface_api: HfApi,
    model_info: ModelInfo,
    avocado_team_members: List[str],
) -> ModelMetadataResult:
    """Analyze metadata for a model; no globals."""
    model_id = model_info.id
    metadata_result = ModelMetadataResult(id=model_id)

    # Ignore GGUFs (keep the simple rule you had)
    if "gguf" in (model_info.tags or []):
        metadata_result.should_skip = True
        metadata_result.metadata_issues.append(MetadataIssues.WITH_GGUF.value)
        logger.info(f"Skipped {model_id} : GGUF")
        return metadata_result

    # Some models do not have a discussion tab
    try:
        discussions = list(huggingface_api.get_repo_discussions(model_id))
    except Exception:
        metadata_result.should_skip = True
        metadata_result.metadata_issues.append(MetadataIssues.NO_DISCUSSION_TAB.value)
        logger.info(f"Skipped {model_id} : No Discussion Tab")
        return metadata_result

    # Check Avocado participation
    discussions_with_avocado = []
    for discussion in discussions:
        if discussion.author in avocado_team_members:
            discussions_with_avocado.append(
                AvocadoDiscussion(
                    title=discussion.title,
                    author=discussion.author,
                    url=f"https://huggingface.co/{model_id}/discussions/{discussion.num}",
                )
            )
    metadata_result.discussions_with_avocado_participation = discussions_with_avocado

    # Metadata issues
    if model_info.library_name is None:
        metadata_result.metadata_issues.append(MetadataIssues.NO_LIBRARY_NAME.value)

    if model_info.pipeline_tag is None:
        metadata_result.metadata_issues.append(MetadataIssues.NO_PIPELINE_TAG.value)

    return metadata_result


if __name__ == "__main__":
    slack_config = SlackConfig()
    config = ModelCheckerConfig()
    huggingface_api = HfApi(token=os.environ["HF_TOKEN"])
    client = WebClient(token=os.environ["SLACK_TOKEN"])

    # 1: Fetch the top N trending models
    trending_models = huggingface_api.list_models(
        sort="trendingScore", limit=config.num_trending_models
    )

    # 2: Process model metadata
    avocado_members = list(getattr(config, "avocado_team_members", []))
    metadata_results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_model_info = {
            executor.submit(
                analyze_model_metadata, huggingface_api, model_info, avocado_members
            ): model_info
            for model_info in trending_models
        }

        for future in as_completed(future_to_model_info):
            model_info = future_to_model_info[future]
            try:
                result = future.result()
                metadata_results.append(asdict(result))  # keep Dataset conversion
            except Exception as e:
                logger.error(f"Error processing model {model_info.id}: {e}")

    trending_models_metadata_ds = Dataset.from_list(metadata_results)

    # 3: Categorize the models by issue (single pass; readable)
    models_by_issue_type: Dict[str, List[Dict]] = {
        issue.value: [] for issue in MetadataIssues
    }
    for row in trending_models_metadata_ds:
        for issue in row["metadata_issues"]:
            if issue in models_by_issue_type:
                models_by_issue_type[issue].append(row)

    # 4: Send the updates to Slack
    today = datetime.now().strftime("%Y-%m-%d")
    messages = [
        SlackMessage(msg_type=SlackMessageType.DIVIDER),
        SlackMessage(
            text=f"Meta Data Report for {today}", msg_type=SlackMessageType.HEADER
        ),
    ]
    send_slack_message(
        client=client, channel_name=slack_config.channel_name, messages=messages
    )

    for issue_type, models in models_by_issue_type.items():
        # Keep your original skip rule for these categories
        if issue_type in ["no_discussion_tab", "with_gguf"]:
            continue

        title_msg = SlackMessage(
            text=f"*{' '.join(issue_type.split('_'))}*",
            msg_type=SlackMessageType.SECTION,
        )

        # Keep your original filtering semantics
        filtered_ids = [
            m["id"]
            for m in models
            if not m["should_skip"] and not m["discussions_with_avocado_participation"]
        ]

        if not filtered_ids:
            send_slack_message(
                client=client,
                channel_name=slack_config.channel_name,
                messages=[
                    title_msg,
                    SlackMessage(
                        text="Nothing today 🤙", msg_type=SlackMessageType.SECTION
                    ),
                ],
            )
            continue

        lines = [_model_link_line(mid) for mid in filtered_ids]
        for chunk in _chunk_markdown(lines, max_len=2900):
            send_slack_message(
                client=client,
                channel_name=slack_config.channel_name,
                messages=[
                    title_msg,
                    SlackMessage(text=chunk, msg_type=SlackMessageType.SECTION),
                ],
            )
