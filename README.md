# Trending Models Checker

A tool that monitors trending models on the Hugging Face Hub and reports metadata issues to Slack.

## Overview

This script fetches the top trending models from Hugging Face, checks their metadata for common issues,
and reports problems to a Slack channel. It also archives the daily trending model data to a HuggingFace dataset repository.

Issues checked:
- Missing library name
- Missing pipeline tag
- Custom code usage
- Missing discussion tab

## Installation

```bash
git clone https://github.com/Vaibhavs10/metadata-sniffer
cd metadata-sniffer

pip install -r requirements.txt

touch .env
```

## Configuration

Create a `.env` file with the following variables:

```sh
SLACK_WEBHOOK_URL=your_slack_webhook_url
```

**Note for Hugging Face team members:** Contact @ariG23498 to get the Slack webhook URL for your `.env` file.

## Usage

```bash
# Basic usage
python trending_checker.py

# Debug mode (prints Slack messages without sending them)
python trending_checker.py --debug

# Limit the number of models checked
python trending_checker.py --limit 50

# Enable verbose logging
python trending_checker.py --verbose
```

## Command Line Arguments

- `-d, --debug`: Run in debug mode without sending Slack messages
- `-l, --limit`: Limit the number of trending models to check (default: 100)
- `--verbose`: Enable detailed logging

## Output

- Saves daily trending model data to `trending_data/{today's date}.json`
- Uploads data to the HuggingFace dataset repository: `ariG23498/trending_models`
- Sends Slack notifications about problematic models

The README has been created with the help of Claude.