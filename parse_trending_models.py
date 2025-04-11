from huggingface_hub import HfApi, file_exists, upload_file
import json
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def send_slack_message(webhook_url, message=None, blocks=None):
    payload = {"text": message or "Trending Model Issues", "blocks": blocks} if blocks else {"text": message}
    response = requests.post(webhook_url, json=payload)
    if response.status_code != 200:
        print(f"Failed to send Slack message: {response.text}")

def fetch_trending_models(hf_api):
    return hf_api.list_models(sort="trendingScore", limit=100)

def check_model_metadata(model, problematic_models):
    if model.library_name is None:
        problematic_models["models_with_no_library_name"].append(model.id)

    if model.pipeline_tag is None:
        problematic_models["models_with_no_pipeline_tag"].append(model.id)

    if "custom_code" in (model.tags or []):
        problematic_models["models_with_custom_code"].append(model.id)

def format_issues_block(problematic_models):
    blocks = []

    def add_section(title, model_ids, chunk_size=15):
        if not model_ids:
            return

        for i in range(0, len(model_ids), chunk_size):
            chunk = model_ids[i:i + chunk_size]
            block_text = f"*{title}* (_{len(model_ids)} models_)\n" if i == 0 else ""
            block_text += "\n".join([f"• <https://huggingface.co/{mid}|{mid}>" for mid in chunk])
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": block_text}
            })
            blocks.append({"type": "divider"})


    blocks.append({
        "type": "header",
        "text": {"type": "plain_text", "text": "🔍 Trending Model Issues Report", "emoji": True}
    })

    add_section("📚 Models without a Library Name", problematic_models["models_with_no_library_name"])
    add_section("🏷️ Models without a Pipeline Tag", problematic_models["models_with_no_pipeline_tag"])
    add_section("🧑‍💻 Models with Custom Code", problematic_models["models_with_custom_code"])

    return {"blocks": blocks}

if __name__ == "__main__":
    data_folder = "trending_data"
    repo_id = "ariG23498/trending_models"
    slack_webhook_url = os.getenv("EXP_SLACK_WEBHOOK_URL")
    os.makedirs(data_folder, exist_ok=True)

    hf_api = HfApi()
    today = datetime.now().strftime("%Y-%m-%d")
    output_file = f"{data_folder}/{today}.json"

    trending_models = fetch_trending_models(hf_api)

    model_data = []
    problematic_models = {
        "models_with_no_library_name": [],
        "models_with_no_pipeline_tag": [],
        "models_with_custom_code": [],
    }

    for model in trending_models:
        model_data.append(model.__dict__)
        check_model_metadata(model, problematic_models)

    with open(output_file, "w") as f:
        json.dump(model_data, f, indent=2, default=str)

    if not file_exists(repo_id=repo_id, filename=output_file, repo_type="dataset"):
        upload_file(
            repo_id=repo_id,
            path_or_fileobj=output_file,
            path_in_repo=output_file,
            repo_type="dataset",
            commit_message=f"Upload {output_file}"
        )
        upload_message = f"Uploaded trending models data: {today}"
        if slack_webhook_url:
            send_slack_message(slack_webhook_url, message=upload_message)

    if slack_webhook_url and any(problematic_models.values()):
        slack_blocks = format_issues_block(problematic_models)
        print(slack_blocks)
        send_slack_message(slack_webhook_url, blocks=slack_blocks["blocks"])

    print("Script completed.")
