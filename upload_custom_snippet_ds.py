from datasets import load_dataset, concatenate_datasets, Dataset

def create_todays_ds():
    model_id_description = {
        "model_id": list(model_id_description.keys()),
        "description": list(model_id_description.values()),
    }
    return Dataset.from_dict(model_id_description)

if __name__ == "__main__":
    # Change this every day
    model_id_description = {
        "jinaai/jina-embeddings-v3": "works with sentence transformers",
        "Alibaba-NLP/gte-Qwen2-7B-instruct": "works with sentence transformers",
        "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0": "works with sentence transformers",
        "deepseek-ai/DeepSeek-R1": "needs specific CUDA version to run",
        "deepseek-ai/DeepSeek-V3-0324": "needs specific CUDA version to run",
        "XiaomiMiMo/MiMo-7B-RL": "works with transformers automodel, but not with pipeline",
        "deepseek-ai/DeepSeek-Prover-V2-671B": "needs specific CUDA version to run",
    }
    today_ds = create_todays_ds()
    orig_ds = load_dataset("model-metadata/model-id-custom-code-check")
    concatenated_ds = concatenate_datasets([orig_ds["train"], today_ds])
    concatenate_datasets.push_to_hub("model-metadata/model-id-custom-code-check")