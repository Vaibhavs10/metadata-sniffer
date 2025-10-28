from dataclasses import dataclass, field
from typing import List


@dataclass
class SlackConfig:
    # channel_name = "#hub-model-metadata-snippets-sprint"
    channel_name = "#exp-slack-alerts"


@dataclass
class DatasetConfig:
    # Top N trending models of the day
    trending_models_metadata_id: str = "model-metadata/trending_models_metadata"

    # Dataset for HF JOBS (using uv)
    code_execution_files_dataset_id: str = "model-metadata/code_execution_files"
    code_python_files_dataset_id: str = "model-metadata/code_python_files"

    # Dataset that holds urls for HF Jobs
    hf_jobs_url_dataset_id = "model-metadata/hf_jobs_url"

    models_executed_with_urls_dataset_id = "model-metadata/models_executed_urls"


@dataclass
class ModelCheckerConfig:
    avocado_team_members: List[str] = field(
        default_factory=lambda: [
            "ariG23498",
            "reach-vb",
            "pcuenq",
            "burtenshaw",
            "davanstrien",
            "merve",
            "sergiopaniego",
            "Steveeeeeeen",
            "nielsr",
            "dn6",
            "linoyts",
            "sayakpaul",
        ]
    )
    num_trending_models: int = 100


@dataclass
class ExecuteCustomCodeConfig:
    docker_image: str = "ghcr.io/astral-sh/uv:debian"
    pattern: str = (
        r"ID:\s*([a-zA-Z0-9]+)\s*View at:\s*(https://huggingface\.co/jobs/[^/]+/\1)"
    )
