# Trending Models Pipeline

## Overview

The pipeline automates the following tasks:

1. Fetches trending models from Hugging Face and checks their metadata for issues (e.g., missing library names, pipeline tags, or discussion tabs).
2. Processes models with custom code by extracting snippets from Jupyter notebooks, wrapping them with error handling, and uploading them to Hugging Face datasets.
3. Executes the processed scripts on Hugging Face jobs with GPU selection based on estimated VRAM requirements.
4. Summarizes execution results and sends detailed reports to Slack, categorizing outcomes as successful, failed, or skipped.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/trending-models-pipeline.git
   cd trending-models-pipeline
   ```

2. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following variables:

   ```env
   HF_TOKEN=your_huggingface_api_token
   SLACK_WEBHOOK_URL=your_slack_webhook_url
   EXP_SLACK_WEBHOOK_URL=your_experiment_slack_webhook_url
   ```

## Usage

To run the pipeline manually, execute the scripts in order:

```bash
python 01_parse_trending_models.py
python 02_process_models_with_custom_code.py
python 03_execute_hf_jobs.py
sleep 900  # Wait 15 minutes
python 04_summarise_custom_code.py
```

For automated execution, configure the GitHub Action workflow (see GitHub Actions).

To run in debug mode (e.g., for `01_parse_trending_models.py`):

```bash
python 01_parse_trending_models.py --debug --verbose
```

## Configuration

Environment variables in `.env`:

- `HF_TOKEN`: Hugging Face API token for authentication.
- `SLACK_WEBHOOK_URL`: Slack webhook for metadata issue notifications.
- `EXP_SLACK_WEBHOOK_URL`: Slack webhook for execution summaries.

Additional configurations (hardcoded, consider externalizing):

- Dataset IDs: `model-metadata/trending_models`, `model-metadata/models_with_custom_code`, etc.
- GPU VRAM mappings: Defined in `03_execute_hf_jobs.py`.
- Slack message limits: Defined in `04_summarise_custom_code.py`.
