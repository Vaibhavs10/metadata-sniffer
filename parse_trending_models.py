from huggingface_hub import HfApi, file_exists, upload_file
import json
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def send_slack_message(webhook_url, message):
    payload = {"text": message}
    requests.post(webhook_url, json=payload)

def fetch_trending_models(hf_api):
    return hf_api.list_models(sort="trendingScore", limit=100)

def check_model_metadata(model):
    issues = []
    if not model.library_name:
        issues.append("Missing library name")
    if not model.pipeline_tag:
        issues.append("Missing pipeline tag")
    if not model.config:
        issues.append("Missing config")
    if "custom_code" in (model.tags or []):
        issues.append("Contains custom code")
    return issues

def format_issues(problematic_models):
    if not problematic_models:
        return "No issues found with trending models."
    
    message = "Found issues with trending models:\n"
    for model in problematic_models:
        message += f"\n*{model['model_name']}* (<{model['repository_url']}>)\n"
        for issue in model['issues']:
            message += f"  - {issue}\n"
    return message

if __name__ == "__main__":
    data_folder = "trending_data"
    repo_id = "ariG23498/trending_models"
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    os.makedirs(data_folder, exist_ok=True)
    
    hf_api = HfApi()
    today = datetime.now().strftime("%Y-%m-%d")
    output_file = f"{data_folder}/{today}.json"

    trending_models = fetch_trending_models(hf_api)
    
    model_data = []
    problematic_models = []
    
    for model in trending_models:
        model_data.append(model.__dict__)
        issues = check_model_metadata(model)
        if issues:
            problematic_models.append({
                "model_name": model.id,
                "repository_url": f"https://huggingface.co/{model.id}",
               "issues": issues,
            })

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(model_data, f, indent=2, default=str)

    # Upload if not exists
    if not file_exists(repo_id=repo_id, filename=output_file, repo_type="dataset"):
        upload_file(
            repo_id=repo_id,
            path_or_fileobj=output_file,
            path_in_repo=output_file,
            repo_type="dataset",
            commit_message=f"Upload {output_file}"
        )
        upload_message = f"Uploaded trending models data: {today}"
        print(upload_message)
        if slack_webhook_url:
            send_slack_message(slack_webhook_url, upload_message)

    # Send Slack notification for issues
    if slack_webhook_url and problematic_models:
        slack_message = format_issues(problematic_models)
        send_slack_message(slack_webhook_url, slack_message)
    
    print("Script completed.")