# from datasets import load_dataset, Dataset

# model_id_description = {
#     "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1" : "too big",
#     "microsoft/bitnet-b1.58-2B-4T" : "needs custom transformers installation",
#     "OpenGVLab/InternVL3-78B" : "too big",
#     "jinaai/jina-reranker-m0" : "works with AutoModel, does not with pipeline",
#     "ServiceNow-AI/Apriel-5B-Instruct": "works with AutoModel, does not with pipeline",
#     "ServiceNow-AI/Apriel-5B-Base" : "works with AutoModel, does not with pipeline",
#     "OpenGVLab/InternVL3-1B" : "works with AutoModel, does not with pipeline",
#     "nvidia/Llama-3_3-Nemotron-Super-49B-v1" : "too big",
#     "briaai/RMBG-2.0" : "needs kornia to run AutoModel, pipeline fails anyways",
#     "Dream-org/Dream-v0-Instruct-7B" : "works",
#     "OpenGVLab/InternVL3-8B" : "works with AutoModel, does not with pipeline",
#     "OpenGVLab/InternVL3-14B" : "works with AutoModel, does not with pipeline",
#     "starvector/starvector-8b-im2svg": "needs custom installation",
#     "microsoft/Phi-4-multimodal-instruct" : "needs backoff python package",
# }

# model_id_description = {
#     "model_id": list(model_id_description.keys()),
#     "description": list(model_id_description.values()),
# }

# ds = Dataset.from_dict(model_id_description)
# ds.push_to_hub("model-metadata/model-id-custom-code-check", private=True)
