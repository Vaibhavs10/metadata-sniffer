from huggingface_hub import HfApi
import os
from datetime import datetime
import json

def fetch_trending_models(hf_api, max_models=100):
    return hf_api.list_models(sort="trendingScore", limit=max_models)

def check_model_metadata(model):
    problems = []
    if model.library_name is None:
        problems.append("Missing library name")
    if model.pipeline_tag is None:
        problems.append("Missing pipeline tag")
    if model.config is None:
        problems.append("Missing config")
    if "custom_code" in model.tags:
        problems.append("Contains custom code")

    model_repo_url = f"https://huggingface.co/{model.id}"
    colab_notebook_url = f"https://colab.research.google.com/#fileId=https%3A//huggingface.co/{model.id}.ipynb"
    
    return {
        "model_name": model.id,
        "repository_url": model_repo_url,
        "colab_url": colab_notebook_url,
        "problems": problems
    }

def serialize_to_string(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

if __name__ == "__main__":
    data_folder = "trending_data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    try:
        huggingface_api = HfApi()
        trending_models = fetch_trending_models(huggingface_api, max_models=10)

        all_model_details = []
        for model in trending_models:
            all_model_details.append(model.__dict__)
            metadata_check = check_model_metadata(model)
            
            if metadata_check["problems"]:
                print(f"Model: {metadata_check['model_name']}")
                print(f"Repository: {metadata_check['repository_url']}")
                print(f"Colab: {metadata_check['colab_url']}")
                print("Problems found:")
                for problem in metadata_check["problems"]:
                    print(f"  - {problem}")

        today = datetime.now().strftime("%Y-%m-%d")
        output_file = f"{data_folder}/{today}.json"
        with open(output_file, "w") as json_file:
            json.dump(all_model_details, json_file, indent=2, default=serialize_to_string)

    except Exception as e:
        print(f"Error occurred: {e}")