import requests
import json
from datetime import datetime
import os
import time

# Create a directory to store the JSON files if it doesn't exist
output_dir = "trending_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get current date for the filename
current_date = datetime.now().strftime("%Y-%m-%d")

# Get Hugging Face token from environment variable or use a default value
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Function to fetch trending models for a specific task
def fetch_trending_models(limit=100):
    url = f"https://huggingface.co/api/models"
    params = {
        "limit": limit,
        "sort": "trending",
        "full": "True",
        "config": "True",
        "direction": -1,
    }
    
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {current_date}: {e}")
        return None

print(f"Fetching trending models for {current_date}")

# Fetch trending models for the current task
trending_models = fetch_trending_models()

if trending_models is not None:
    # Create filename with task and date
    filename = f"{output_dir}/{current_date}.json"
    
    # Save the data to a JSON file
    with open(filename, 'w') as f:
        json.dump(trending_models, f, indent=2)
    
    print(f"Saved trending models for {current_date} to {filename}")
    
    # Add a small delay to avoid hitting rate limits
    time.sleep(0.5)

print("All tasks completed!")