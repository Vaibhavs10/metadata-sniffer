from dataclasses import dataclass, field
from typing import List


@dataclass
class SlackConfig:
    channel_name = "#hub-model-metadata-snippets-sprint"


@dataclass
class DatasetConfig:
    models_with_custom_code_dataset_id: str = "model-metadata/models_with_custom_code"
    custom_code_py_files_dataset_id: str = "model-metadata/custom_code_py_files"
    custom_code_execution_files_dataset_id: str = (
        "model-metadata/custom_code_execution_files"
    )
    model_vram_code_dataset_id = "model-metadata/model_vram_code"
    models_executed_with_urls_dataset_id = "model-metadata/models_executed_urls"


@dataclass
class ModelCheckerConfig:
    avocado_team_members: List[str] = field(
        default_factory=lambda: [
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
    )
    num_trending_models: int = 100


@dataclass
class ExecuteCustomCodeConfig:
    docker_image: str = "ghcr.io/astral-sh/uv:debian"
    pattern: str = (
        r"ID:\s*([a-zA-Z0-9]+)\s*View at:\s*(https://huggingface\.co/jobs/[^/]+/\1)"
    )
