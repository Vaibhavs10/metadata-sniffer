# Trending Models Metadata Sniffer

A Python script to fetch, validate, and save trending models from Hugging Face.

## Purpose

1. Fetches the top trending models from Hugging Face.
2. Checks model metadata for issues (e.g., missing library name, custom code).
3. Saves all model details to a JSON file in trending_data/ with the current date.

## Usage

```bash
python parse_trending_models.py
```

Outputs metadata problems to the terminal if found. Saves model data to `trending_data/YYYY-MM-DD.json`.