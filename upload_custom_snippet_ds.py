from datasets import Dataset, concatenate_datasets, load_dataset


def create_todays_ds():
    # Change this every day
    model_id_description = {
        "openbmb/AgentCPM-GUI": "does not have a code snippet section",
        "microsoft/Magma-8B": "needs open clip package to be installed",
        "suayptalha/Arcana-Qwen3-2.4B-A0.6B": "user wants people to snapshot download the repo, hence used relative classes in config",
    }
    model_id_description = {
        "model_id": list(model_id_description.keys()),
        "description": list(model_id_description.values()),
    }
    return Dataset.from_dict(model_id_description)


if __name__ == "__main__":
    today_ds = create_todays_ds()
    orig_ds = load_dataset("model-metadata/model-id-custom-code-check")
    concatenated_ds = concatenate_datasets([orig_ds["train"], today_ds])
    concatenated_ds.push_to_hub("model-metadata/model-id-custom-code-check")
